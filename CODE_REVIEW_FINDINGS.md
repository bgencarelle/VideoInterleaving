# Code Review Findings - Wonky/Weird/Leftover Issues

## 🔴 Critical Issues

### 1. **Duplicate Monitor Services**
**Files**: `monitor_service.py` vs `multimonitor.py`

**Issue**: Two similar monitor services exist:
- `monitor_service.py` - Old unified monitor (reads from `stats_*.json` files)
- `multimonitor.py` - New independent monitor (checks ports/HTTP)

**Recommendation**: 
- Decide which one to keep (probably `multimonitor.py` since it's newer and more comprehensive)
- Remove or deprecate `monitor_service.py`
- Or merge functionality if both are needed

### 2. **Hardcoded IP Addresses**
**Files**: `index_server.py`, `index_client.py`

**Issue**: Hardcoded IPs that won't work on other networks:
```python
# index_server.py line 47
start_server = websockets.serve(handle_client, "192.168.178.23", 12345)

# index_client.py line 21
async def receive_midi_data(uri="ws://192.168.178.23:12345"):
```

**Recommendation**: 
- Use environment variables or config file
- Or detect network interface automatically
- Or make it a command-line argument

### 3. **Unused Import**
**File**: `monitor_service.py` line 7

**Issue**: `import glob` is imported but never used

**Recommendation**: Remove the import

---

## 🟡 Medium Issues

### 4. **Commented Out Code**
**Files**: Multiple

**Issues**:
- `image_display.py` line 9: `#import webp  # Strict Requirement: pip install webp`
- `midi_control.py` lines 6-8: Commented backend imports
- `libwebp_loader.py` line 93: Commented eager load option
- `folder_selector.py` line 6: Commented import

**Recommendation**: 
- Remove commented code if not needed
- Or add TODO comments explaining why it's commented

### 5. **Bare Exception Handlers**
**Files**: Multiple (92 instances found)

**Issues**: Many `except:` or `except Exception:` without specific handling

**Examples**:
- `web_service.py` line 307: `except:`
- `monitor_service.py` line 149: `except:`

**Recommendation**: 
- Use specific exception types where possible
- At minimum, log the exception for debugging

### 6. **Duplicate Dictionary Definitions**
**Files**: `globals.py`, `index_client.py`, `midi_control.py`

**Issue**: Same `midi_data_dictionary` structure defined in multiple places:
```python
midi_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    ...
}
```

**Recommendation**: 
- Centralize in one place (maybe `midi_control.py`)
- Import from there

### 7. **Files with Issues (But Still Used)**
**Files**: `globals.py`, `index_server.py`/`index_client.py`, `index_calculator.py`

**Issue**: 
- `globals.py` IS used (by `folder_selector.py`, `index_calculator.py`) but defines dictionaries duplicated elsewhere
- `index_server.py`/`index_client.py` are used for MIDI sync but have hardcoded IPs
- `index_calculator.py` IS used (by `image_display.py`) for clock/index calculation

**Recommendation**: 
- Keep `globals.py` but consolidate duplicate dictionary definitions
- Fix hardcoded IPs in `index_server.py`/`index_client.py` (make configurable)
- These files are part of the MIDI sync system, so they're needed

### 8. **Outdated Comments**
**Files**: Multiple

**Issues**:
- `web_service.py` lines 275, 319: `# [NEW] Dynamic Template Fallback` - not new anymore
- `web_service.py` line 351: `# [NEW] Stall Timeout Logic` - not new anymore
- `main.py` line 13: `# [CHANGE] Updated reserved ports` - change is done
- `main.py` line 26: `# [FIX] Allow reusing the address` - fix is done

**Recommendation**: Remove `[NEW]`, `[CHANGE]`, `[FIX]` markers or update to current status

### 9. **Inconsistent Comment Style**
**Files**: Multiple

**Issue**: Mix of comment styles:
- Some files use `#` for section headers
- Some use `# --- SECTION ---`
- Some use `# SECTION:`

**Recommendation**: Standardize comment style (prefer `# --- SECTION ---`)

### 10. **Dead Code in server_config.py**
**File**: `server_config.py` line 134

**Issue**: Comment says "Note: ASCIIWEB monitor uses 1980, but we'll use the web monitor (1978) for that"

**Recommendation**: 
- This is confusing - clarify or remove
- In "all" mode, we use 1978 for web monitor, but ASCIIWEB mode uses 1980
- The comment might be outdated

---

## 🟢 Minor Issues

### 11. **Duplicate Port Constants**
**Files**: `server_config.py`, `multimonitor.py`

**Issue**: `multimonitor.py` defines `MODE_PORTS` dictionary that duplicates defaults from `server_config.py`

**Recommendation**: 
- Import defaults from `server_config.py` instead of duplicating
- Or make `server_config.py` export a function to get default ports

### 12. **Inconsistent Error Messages**
**Files**: Multiple

**Issue**: Some errors use emoji (⚠️, ❌), some don't

**Recommendation**: Standardize error message format

### 13. **Unused Variables/Imports**
**Files**: Multiple

**Issues**:
- Check for any unused imports (already found `glob` in `monitor_service.py`)
- Check for unused variables

**Recommendation**: Run a linter or static analysis tool

### 14. **Magic Numbers**
**Files**: Multiple

**Issues**: Some magic numbers still exist:
- `monitor_service.py` line 144: `if time.time() - mtime > 5:` (5 second timeout)
- Various timeout values scattered around

**Recommendation**: Extract to named constants

### 15. **Inconsistent Naming**
**Files**: Multiple

**Issues**:
- Some functions use `snake_case`, some might use different patterns
- Some constants use `UPPER_CASE`, some use different patterns

**Recommendation**: Enforce consistent naming (Python standard: `snake_case` for functions, `UPPER_CASE` for constants)

---

## 📋 Summary of Recommendations

### High Priority (Do Now)
1. ✅ Remove unused `import glob` from `monitor_service.py`
2. ✅ Decide on `monitor_service.py` vs `multimonitor.py` (keep one, remove/deprecate other)
3. ✅ Fix hardcoded IPs in `index_server.py` and `index_client.py`
4. ✅ Remove `[NEW]`, `[CHANGE]`, `[FIX]` markers from comments

### Medium Priority (Do Soon)
5. Clean up commented-out code (remove or document why it's there)
6. Replace bare `except:` with specific exception types
7. Consolidate duplicate dictionary definitions (keep `globals.py` but deduplicate)
8. Fix hardcoded IPs in `index_server.py`/`index_client.py` (make configurable)

### Low Priority (Nice to Have)
9. Standardize comment style
10. Extract magic numbers to constants
11. Standardize error message format
12. Run linter to find unused imports/variables

---

## 🔍 Files Status

1. **`globals.py`** - ✅ USED (by `folder_selector.py`, `index_calculator.py`) - Keep but deduplicate
2. **`index_server.py`** / **`index_client.py`** - ✅ USED (MIDI sync system) - Fix hardcoded IPs
3. **`monitor_service.py`** - ⚠️ DUPLICATE of `multimonitor.py`? - Decide which to keep
4. **`index_calculator.py`** - ✅ USED (by `image_display.py`) - Clock/index calculation

---

## ✅ What's Good

- Code is generally well-structured
- Good separation of concerns
- Proper error handling in most places
- Good use of threading and async patterns
- Security considerations (path traversal protection, localhost defaults)

