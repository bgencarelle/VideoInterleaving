# calculators.py Analysis Report

## Executive Summary

This document categorizes all code in `calculators.py` into three categories:
1. **✅ ACTIVELY USED** - Code that is currently used and essential
2. **⚠️ POTENTIALLY USED** - Code that may be used in certain conditions/legacy paths
3. **❌ DEAD CODE** - Code that is never used and should be removed

---

## ✅ ACTIVELY USED CODE (Keep)

### 1. `init_all(clock_mode)` - **CRITICAL**
- **Location**: Lines 168-185
- **Called by**:
  - `image_display.py:209` (main entry point - always called)
  - `utilities/csv_list_maker.py:82` (utility script)
- **Purpose**: Core initialization function that loads image lists and sets up frame duration
- **Returns**: `(csv_main, main_image_paths, float_image_paths)`
- **Status**: **KEEP** - Essential for application startup

### 2. `select_img_list_files()` - **USED**
- **Location**: Lines 104-141
- **Called by**: `init_all()` line 172
- **Purpose**: Selects CSV files containing image paths from `GENERATED_LISTS_DIR`
- **Uses**: `settings.GENERATED_LISTS_DIR` (dynamic, respects mode)
- **Status**: **KEEP** - Required by `init_all()`

### 3. `get_image_names_from_csv(file_path)` - **USED**
- **Location**: Lines 144-150
- **Called by**: `init_all()` lines 173-174
- **Purpose**: Reads image paths from CSV file
- **Side effect**: Sets global `png_paths_len`
- **Note**: **DUPLICATE** - Same function exists in `utilities/csv_list_maker.py:9`
- **Status**: **KEEP** - But should remove duplicate in `csv_list_maker.py`

### 4. `calculate_index(frame_counter)` - **USED (MIDI modes only)**
- **Location**: Lines 153-165
- **Called by**: `midi_control.py:134, 275, 278` (MIDI clock modes only)
- **Purpose**: Calculates image index based on frame counter for MIDI clock modes
- **Used in modes**: `MTC_CLOCK`, `MIDI_CLOCK`, `MIXED_CLOCK`
- **Note**: Replaced by `index_calculator.py` for `FREE_CLOCK` mode
- **Status**: **KEEP** - Required for MIDI clock modes

### 5. `FREE_CLOCK` constant - **USED (but duplicate)**
- **Location**: Line 14
- **Value**: `255`
- **Also defined**: `constantStorage/midi_constants.py:9` (duplicate!)
- **Used in**: `init_all()` line 177, `midi_constants.py`
- **Status**: **KEEP** - But should remove duplicate, import from `midi_constants.py`

### 6. Global `png_paths_len` - **USED**
- **Location**: Line 8 (initialized to 2221)
- **Set by**: `get_image_names_from_csv()` line 149
- **Used by**: `calculate_index()` line 156, 164
- **Status**: **KEEP** - Required by `calculate_index()` for MIDI modes

### 7. Global `frame_duration` - **USED (MIDI modes)**
- **Location**: Line 9 (initialized to 4.0)
- **Set by**: `calculate_frame_duration()` line 99 or `init_all()` line 179
- **Used by**: `calculate_index()` line 155
- **Status**: **KEEP** - Required by `calculate_index()` for MIDI modes

### 8. Global `video_length` - **USED (MIDI modes)**
- **Location**: Line 10 (initialized to 9173)
- **Set by**: `calculate_frame_duration()` or `init_all()` line 178
- **Status**: **KEEP** - Used in frame duration calculation

---

## ⚠️ POTENTIALLY USED CODE (Conditional/Legacy)

### 1. `calculate_frame_duration()` - **CONDITIONAL**
- **Location**: Lines 58-101
- **Called by**: 
  - `init_all()` line 183 (only if `clock_mode != FREE_CLOCK`)
  - `get_video_length()` line 39 (if CSV doesn't exist)
- **Purpose**: Interactive setup for video length and frame duration
- **Triggers**: User input prompts (requires terminal interaction)
- **Dependencies**: 
  - `presets/set_video_length.csv` file
  - `mido` library (for MIDI mode)
- **Status**: **MIGHT BE USED** - Only executes when:
  - `clock_mode != FREE_CLOCK` (default is `FREE_CLOCK`)
  - OR when `get_video_length()` is called and CSV doesn't exist
- **Finding**: Default `CLOCK_MODE = FREE_CLOCK`, so this path is rarely executed
- **Recommendation**: Keep but document as legacy/optional feature

### 2. `get_video_length(video_number=0)` - **CONDITIONAL**
- **Location**: Lines 34-55
- **Called by**: `calculate_frame_duration()` line 98
- **Purpose**: Reads video length from `presets/set_video_length.csv`
- **Dependencies**: `presets/set_video_length.csv` file must exist
- **Status**: **MIGHT BE USED** - Only if:
  - `calculate_frame_duration()` is called
  - AND `presets/set_video_length.csv` exists
- **Finding**: No `presets/` folder found in project root
- **Recommendation**: Keep but likely unused in practice

### 3. `set_video_length(video_name, video_name_length)` - **CONDITIONAL**
- **Location**: Lines 24-31
- **Called by**: `calculate_frame_duration()` line 96
- **Purpose**: Writes video length to `presets/set_video_length.csv`
- **Status**: **MIGHT BE USED** - Only during initial setup via `calculate_frame_duration()`
- **Finding**: Creates `presets/` folder if it doesn't exist
- **Recommendation**: Keep but likely unused in practice

### 4. `get_midi_length(midi_file_path)` - **CONDITIONAL**
- **Location**: Lines 17-21
- **Called by**: `calculate_frame_duration()` line 82 (if user chooses MIDI mode)
- **Purpose**: Calculates video length from MIDI file duration
- **Dependencies**: `mido` library
- **Status**: **MIGHT BE USED** - Only if:
  - `calculate_frame_duration()` is called
  - AND user chooses 'd' (MIDI-derived) mode
- **Recommendation**: Keep but likely unused in practice

### 5. Hardcoded default values - **FALLBACKS**
- **Location**: Lines 8-10
  - `png_paths_len = 2221`
  - `frame_duration = 4.0`
  - `video_length = 9173`
- **Purpose**: Fallback values if functions never called
- **Status**: **MIGHT BE USED** - Only as fallback if code path fails
- **Finding**: All are overwritten by `init_all()` or `get_image_names_from_csv()`
- **Recommendation**: Keep as safety fallbacks, but document

---

## ❌ DEAD CODE (Remove)

### 1. `bpm_smoothing_window = 10` - **NEVER USED**
- **Location**: Line 11
- **Status**: **DEAD CODE** - No references found in entire codebase
- **Action**: **REMOVE**

### 2. `main()` function - **LEGACY/TEST**
- **Location**: Lines 188-190
- **Purpose**: Test function that runs `init_all(FREE_CLOCK)` if script executed directly
- **Status**: **LEGACY** - Only runs if `python calculators.py` is executed
- **Usage**: No evidence of being called in production
- **Action**: **REMOVE** or document as test/debug function

---

## Key Findings

### 1. Duplicate Code Issues

#### `FREE_CLOCK` constant (duplicate)
- **Defined in**: `calculators.py:14` and `constantStorage/midi_constants.py:9`
- **Impact**: Potential inconsistency
- **Recommendation**: Remove from `calculators.py`, import from `midi_constants.py`

#### `get_image_names_from_csv()` (duplicate)
- **Defined in**: `calculators.py:144` and `utilities/csv_list_maker.py:9`
- **Impact**: Code duplication, maintenance burden
- **Recommendation**: Remove from `csv_list_maker.py`, import from `calculators.py`

### 2. Architecture Observations

#### Clock Mode Usage
- **Default**: `CLOCK_MODE = FREE_CLOCK` (from `midi_constants.py`)
- **Impact**: `calculate_frame_duration()` path rarely executes
- **Finding**: Most code paths use `FREE_CLOCK`, making presets system largely unused
- **Question**: Are non-FREE_CLOCK modes actually used in production?

#### Presets System
- **Location**: `presets/set_video_length.csv`
- **Status**: Folder doesn't exist in project
- **Impact**: `get_video_length()` and `set_video_length()` are effectively dead code
- **Question**: Is the presets CSV system still needed?

#### Index Calculation Split
- **`calculate_index()`**: Used by MIDI modes (`MTC_CLOCK`, `MIDI_CLOCK`, `MIXED_CLOCK`)
- **`index_calculator.py`**: Handles `FREE_CLOCK` mode
- **Impact**: Logic split between two files
- **Question**: Should `calculate_index()` be moved to `midi_control.py` since it's only used there?

### 3. Global Variable Dependencies

#### `png_paths_len`
- **Set by**: `get_image_names_from_csv()` (line 149)
- **Used by**: `calculate_index()` (lines 156, 164)
- **Status**: Required for MIDI clock modes
- **Note**: Also used in `image_display.py` but set locally, not from `calculators.py`

#### `frame_duration`
- **Set by**: `calculate_frame_duration()` or `init_all()`
- **Used by**: `calculate_index()` (line 155)
- **Status**: Required for MIDI clock modes
- **Note**: Also defined in `midi_control.py:19` (separate instance)

---

## Recommendations

### High Priority (Immediate Action)

1. **Remove `bpm_smoothing_window`** (line 11)
   - Dead code, no references
   - **Action**: Delete line 11

2. **Remove duplicate `FREE_CLOCK` definition**
   - Keep in `midi_constants.py`, remove from `calculators.py`
   - **Action**: Remove line 14, add `from constantStorage.midi_constants import FREE_CLOCK`

3. **Remove or document `main()` function**
   - Legacy test function
   - **Action**: Remove lines 188-194 OR add comment: `# Test/debug function - not used in production`

### Medium Priority (Investigate First)

4. **Investigate non-FREE_CLOCK mode usage**
   - Check if `MTC_CLOCK`, `MIDI_CLOCK`, `MIXED_CLOCK` are actually used
   - **Action**: Search logs/configs for non-FREE_CLOCK usage
   - **If unused**: Consider deprecating `calculate_frame_duration()` and presets system

5. **Consolidate duplicate `get_image_names_from_csv()`**
   - Remove from `utilities/csv_list_maker.py`
   - **Action**: Import from `calculators.py` instead

6. **Consider moving `calculate_index()` to `midi_control.py`**
   - Only used by MIDI modes
   - **Action**: Move function to `midi_control.py` if it's the only user

### Low Priority (Cleanup)

7. **Document hardcoded defaults**
   - Add comments explaining fallback values
   - **Action**: Add docstring explaining these are fallback values

8. **Consider deprecating presets system**
   - If FREE_CLOCK is always used, presets CSV is unnecessary
   - **Action**: Add deprecation warning or remove if confirmed unused

---

## Usage Statistics

### Function Call Counts
- `init_all()`: **2** calls (image_display.py, csv_list_maker.py)
- `calculate_index()`: **3** calls (all in midi_control.py)
- `calculate_frame_duration()`: **0** direct calls (only via `init_all()` when `clock_mode != FREE_CLOCK`)
- `get_video_length()`: **0** direct calls (only via `calculate_frame_duration()`)
- `set_video_length()`: **0** direct calls (only via `calculate_frame_duration()`)
- `get_midi_length()`: **0** direct calls (only via `calculate_frame_duration()`)
- `select_img_list_files()`: **2** calls (via `init_all()`)
- `get_image_names_from_csv()`: **4** calls (2 in calculators.py, 2 in csv_list_maker.py duplicate)

### Clock Mode Distribution
- **Default**: `FREE_CLOCK` (255)
- **Non-FREE_CLOCK modes**: `MTC_CLOCK` (0), `MIDI_CLOCK` (1), `MIXED_CLOCK` (2)
- **Finding**: Default is `FREE_CLOCK`, making presets system rarely used

---

## Conclusion

The `calculators.py` file contains:
- **8 actively used functions/constants** (essential)
- **5 potentially used functions** (conditional/legacy paths)
- **2 dead code items** (should be removed)

The file is central to initialization but contains legacy code from when the project used a different clock system. Most conditional code paths are rarely executed due to `FREE_CLOCK` being the default mode.

**Recommended cleanup priority**: Remove dead code first, then investigate if conditional paths are needed, then consolidate duplicates.

