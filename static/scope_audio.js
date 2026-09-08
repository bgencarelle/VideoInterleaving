    // Select a destination without creating or starting an audio context.
    // Keep it for subsequent starts on this page, including after Stop audio.
    let selectedSinkId = null;
    let audioBusy = false;
    let outputCheckPending = false;

    function setAudioBusy(busy) {
      audioBusy = busy;
      for (const id of ["local-btn", "sink-btn", "sink-select"])
        document.getElementById(id).disabled = busy;
      if (!busy && outputCheckPending) void checkSelectedOutput();
    }

    async function applyOutput(id, label) {
      if (actx) await actx.setSinkId(id);
      selectedSinkId = id;
      document.getElementById("output-label").textContent = "Output: " + label;
      document.getElementById("sink-status").textContent = "Output: " + label;
    }

    async function chooseOutput() {
      if (audioBusy) return;
      const st = document.getElementById("sink-status");
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass || !("setSinkId" in AudioContextClass.prototype)) {
        st.textContent = "This browser cannot redirect Web Audio output — change your system default before starting audio.";
        return;
      }
      setAudioBusy(true);
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.selectAudioOutput) {
          const dev = await navigator.mediaDevices.selectAudioOutput();
          await applyOutput(dev.deviceId, dev.label || dev.deviceId);
          return;
        }
        // List only outputs already exposed to this page. Never request input.
        const media = navigator.mediaDevices;
        const outputs = media && typeof media.enumerateDevices === "function"
          ? (await media.enumerateDevices()).filter(d =>
              d.kind === "audiooutput" && d.deviceId) : [];
        const sel = document.getElementById("sink-select");
        sel.textContent = "";
        sel.style.display = outputs.length ? "" : "none";
        if (!outputs.length) {
          st.textContent = "No selectable outputs are exposed to this page. Choose your output in system sound settings before starting audio.";
          return;
        }
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Choose an output…";
        placeholder.disabled = true;
        placeholder.selected = true;
        sel.appendChild(placeholder);
        outputs.forEach((device, index) => {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = device.label || (device.deviceId === "default"
            ? "System default" : "Audio output " + (index + 1));
          sel.appendChild(option);
        });
        sel.value = outputs.some(d => d.deviceId === selectedSinkId)
          ? selectedSinkId : "";
        sel.onchange = async () => {
          if (audioBusy) return;
          setAudioBusy(true);
          try {
            await applyOutput(sel.value, sel.options[sel.selectedIndex].textContent);
          } catch (e) {
            sel.value = selectedSinkId === null ? "" : selectedSinkId;
            st.textContent = "Could not select output: " + e.message;
          } finally { setAudioBusy(false); }
        };
        st.textContent = "Choose an available output. No microphone access is requested.";
      } catch (e) {
        st.textContent = "Output selection failed (" + e.name + "). Try again or choose the system default before starting.";
      } finally { setAudioBusy(false); }
    }

    async function toggleLocal() {
      if (audioBusy) return;
      setAudioBusy(true);
      try {
        if (localRunning) {
          localRunning = false;
          stopLumaLoop();
          if (actx) await actx.close();
          actx = null;
          worklet = null;
          lumImg.removeAttribute("src");
          document.getElementById("local-btn").textContent = "Start audio";
          document.getElementById("local-status").textContent = "Not running.";
          return;
        }
        if (selectedSinkId && selectedSinkId !== "default") {
          const devices = await navigator.mediaDevices.enumerateDevices();
          if (!devices.some(d => d.kind === "audiooutput" && d.deviceId === selectedSinkId))
            throw new Error("Selected output unavailable. Choose an output again.");
        }
        actx = new (window.AudioContext || window.webkitAudioContext)();
        // No source is connected until the chosen destination is applied.
        // A missing/denied device fails startup rather than playing elsewhere.
        if (selectedSinkId !== null) await actx.setSinkId(selectedSinkId);
        await actx.resume();
        const url = URL.createObjectURL(new Blob([WORKLET_SRC], {type: "text/javascript"}));
        try { await actx.audioWorklet.addModule(url); }
        finally { URL.revokeObjectURL(url); }
        worklet = new AudioWorkletNode(actx, "scope-out", {outputChannelCount: [2]});
        worklet.connect(actx.destination);
        lumCanvas = document.createElement("canvas");
        lumCtx = lumCanvas.getContext("2d", {willReadFrequently: true});
        lumImg.src = "scope/luma.mjpg?t=" + Date.now();
        localRunning = true;
        startLumaLoop();
        document.getElementById("local-btn").textContent = "Stop audio";
        setStatus();
      } catch (e) {
        localRunning = false;
        stopLumaLoop();
        if (worklet) worklet.disconnect();
        if (actx) { try { await actx.close(); } catch (_) {} }
        actx = null;
        worklet = null;
        lumImg.removeAttribute("src");
        document.getElementById("local-btn").textContent = "Start audio";
        document.getElementById("local-status").textContent = "Failed: " + e;
      } finally { setAudioBusy(false); }
    }

    // Coalesce device changes while an operation is in flight, then check
    // again when it releases the lock. Never discard a disconnect event.
    async function checkSelectedOutput() {
      outputCheckPending = true;
      if (audioBusy) return;
      outputCheckPending = false;
      if (!selectedSinkId || selectedSinkId === "default") return;
      setAudioBusy(true);
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (devices.some(d => d.kind === "audiooutput" && d.deviceId === selectedSinkId)) return;
        localRunning = false;
        stopLumaLoop();
        if (worklet) worklet.disconnect();
        if (actx) { try { await actx.close(); } catch (_) {} }
        actx = null;
        worklet = null;
        lumImg.removeAttribute("src");
        document.getElementById("local-btn").textContent = "Start audio";
        document.getElementById("local-status").textContent = "Stopped: selected output disconnected.";
        document.getElementById("output-label").textContent = "Selected output unavailable";
        document.getElementById("sink-status").textContent = "Choose an output again before starting.";
      } catch (e) {
        document.getElementById("sink-status").textContent = "Could not check audio outputs: " + e;
      } finally { setAudioBusy(false); }
    }

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener)
      navigator.mediaDevices.addEventListener("devicechange", checkSelectedOutput);
