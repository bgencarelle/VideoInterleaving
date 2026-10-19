/* Latest-only multipart image delivery. No native MJPEG presentation queue. */
(function (root) {
    'use strict';
    const MAX_IMAGE = 16 * 1024 * 1024, MAX_HEADERS = 16384;
    class LiveImageStream {
        constructor(image, options = {}) {
            this.image = image;
            this.onError = options.onError || (() => {});
            this.session = null;
            this.displayUrl = null;
        }
        start(url) {
            this.stop(false);
            const session = {controller: new AbortController(), pending: null, decoding: false};
            this.session = session;
            this.read(url, session);
        }
        stop(clear = true) {
            const previous = this.session;
            this.session = null;
            if (previous) { previous.pending = null; previous.controller.abort(); }
            if (clear) {
                this.image.removeAttribute('src');
                if (this.displayUrl) URL.revokeObjectURL(this.displayUrl);
                this.displayUrl = null;
            }
        }
        async read(url, session) {
            let reader;
            try {
                const response = await fetch(url, {cache: 'no-store', signal: session.controller.signal});
                if (!response.ok || !response.body) throw Error('Image stream unavailable');
                reader = response.body.getReader();
                let buffered = new Uint8Array(0), length = null, mime = null;
                while (this.session === session) {
                    const {value, done} = await reader.read();
                    if (done) throw Error('Image stream ended');
                    if (this.session !== session) break;
                    const combined = new Uint8Array(buffered.length + value.length);
                    combined.set(buffered); combined.set(value, buffered.length); buffered = combined;
                    while (this.session === session) {
                        if (length === null) {
                            let end = -1;
                            for (let i = 0; i + 3 < buffered.length; i++) {
                                if (buffered[i] === 13 && buffered[i+1] === 10 &&
                                    buffered[i+2] === 13 && buffered[i+3] === 10) { end = i; break; }
                            }
                            if (end < 0) {
                                if (buffered.length > MAX_HEADERS) throw Error('Image headers too large');
                                break;
                            }
                            if (end > MAX_HEADERS) throw Error('Image headers too large');
                            const headers = new TextDecoder().decode(buffered.subarray(0, end));
                            const size = headers.match(/(?:^|\r\n)Content-Length:\s*(\d+)\s*(?:\r\n|$)/i);
                            const type = headers.match(/(?:^|\r\n)Content-Type:\s*(image\/(?:jpeg|webp))\s*(?:\r\n|$)/i);
                            length = size ? Number(size[1]) : 0;
                            if (!Number.isSafeInteger(length) || length < 1 || length > MAX_IMAGE || !type)
                                throw Error('Invalid multipart image headers');
                            mime = type[1].toLowerCase();
                            buffered = buffered.subarray(end + 4);
                        }
                        if (buffered.length < length) break;
                        session.pending = new Blob([buffered.slice(0, length)], {type: mime});
                        buffered = buffered.subarray(length); length = null;
                        this.decodeLatest(session);
                    }
                    if (buffered.length > MAX_IMAGE + MAX_HEADERS) throw Error('Image buffer too large');
                }
            } catch (error) {
                if (this.session === session) {
                    this.session = null; session.pending = null; session.controller.abort();
                    this.onError(error);
                }
            } finally {
                if (reader) { try { await reader.cancel(); } catch (_) {} }
            }
        }
        async decodeLatest(session) {
            if (session.decoding) return;
            session.decoding = true;
            try {
                while (this.session === session && session.pending) {
                    const blob = session.pending; session.pending = null;
                    const url = URL.createObjectURL(blob);
                    let retained = false;
                    try {
                        const image = new Image();
                        const loaded = new Promise((resolve, reject) => {
                            image.onload = resolve; image.onerror = () => reject(Error('Image decode failed'));
                        });
                        image.src = url;
                        await loaded;
                        if (this.session !== session) continue;
                        const previous = this.displayUrl;
                        this.image.src = url; this.displayUrl = url; retained = true;
                        if (previous) URL.revokeObjectURL(previous);
                    } finally { if (!retained) URL.revokeObjectURL(url); }
                }
            } catch (error) {
                if (this.session === session) {
                    this.session = null; session.pending = null; session.controller.abort();
                    this.onError(error);
                }
            } finally { session.decoding = false; }
        }
    }
    root.LiveImageStream = LiveImageStream;
    if (typeof module === 'object' && module.exports) module.exports = LiveImageStream;
})(typeof window !== 'undefined' ? window : globalThis);
