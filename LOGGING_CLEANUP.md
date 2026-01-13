# Terminal Logging Cleanup

## Problem
Excessive debug logs were cluttering the terminal output, showing internal OCR details like:
- `🔍 Chunk 8 OCR response attributes: ['construct', 'copy', 'dict', 'document_annotation', ...]`
- `🔍 Chunk 8 has document_annotation: True`
- `🔍 Chunk 8 document_annotation is truthy: True`
- `🔍 Chunk 8 raw document_annotation type: <class 'str'>`
- `🔍 Chunk 8 document_annotation (first 500 chars): ...`

## Solution
Changed verbose diagnostic logs from `logger.info()` to `logger.debug()` or removed them entirely.

## Changes Made

### File: `pdf_processing.py`

**Removed:**
- OCR response attribute inspection logs
- Document annotation type checking logs
- JSON snippet previews (first 500 chars)
- Document annotation keys listing
- DrugInformation count logs
- Page markdown preview logs

**Changed to DEBUG level:**
- JSON decode error messages (only show when debugging is enabled)
- JSON repair status messages

**Kept at INFO level:**
- Chunk completion messages: `"Chunk X complete: Y drugs"`
- Error messages for actual failures

## Log Levels Explained

| Level | When Shown | Purpose |
|-------|-----------|---------|
| **ERROR** | Always | Critical failures that stop processing |
| **WARNING** | Always | Issues that don't stop processing |
| **INFO** | Always (default) | Normal progress updates |
| **DEBUG** | Only with `--debug` flag | Detailed diagnostic information |

## Before vs After

### Before (Cluttered)
```
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 OCR response attributes: ['construct', 'copy', ...]
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 has document_annotation: True
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 document_annotation is truthy: True
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 raw document_annotation type: <class 'str'>
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 document_annotation (first 500 chars): {"DrugInformation":[...
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 document_annotation keys: dict_keys(['DrugInformation', 'FormularyAbbreviations'])
2026-01-13 12:40:14 [INFO] 🔍 Chunk 8 DrugInformation count: 42
2026-01-13 12:40:15 [INFO] Chunk 8 complete: 42 drugs
```

### After (Clean)
```
2026-01-13 12:40:15 [INFO] Chunk 8 complete: 42 drugs
2026-01-13 12:40:16 [INFO] Chunk 9 complete: 38 drugs
2026-01-13 12:40:17 [INFO] Chunk 10 complete: 45 drugs
```

## Enabling Debug Logs (If Needed)

To see the detailed diagnostic logs during troubleshooting, change the logging level in `config.py`:

```python
# config.py (line ~11-14)
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

Or add a command-line flag:
```bash
python main.py --debug
```

## Remaining INFO Logs

These important logs are still shown:
- ✅ Chunk completion: `"Chunk X complete: Y drugs"`
- ✅ Progress updates: `"Progress: X/Y chunks (Z%) | Drugs so far: N"`
- ✅ Final summary: `"Total drugs extracted: X"`
- ✅ Database operations: `"Inserted X drug records for plan Y"`
- ✅ Warnings/Errors: Any issues that need attention

## Impact

**Terminal Output Reduction:** ~80% fewer log lines during normal processing
**Debugging Capability:** Still available via DEBUG level
**Important Information:** Still visible (progress, completion, errors)
