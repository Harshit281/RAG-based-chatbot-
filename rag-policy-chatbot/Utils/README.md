# Utils — Shared Utility & Helper Scripts

This directory contains utility modules used across the RAG pipeline for **text processing** and **terminal output formatting**. All files are Python-only (`.py`) — no notebooks.

---

## Modules

### `text_cleaning.py` — Text Normalisation & Extraction

Pure-function helpers for cleaning raw CSV text before embedding and display. Used primarily by `Scripts/chunk_data.py`.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `clean_text` | `(text: str) → str` | Strips leading/trailing whitespace and collapses internal runs of whitespace into a single space. Returns `""` for non-string inputs. |
| `extract_ref_code` | `(text: str) → str` | Searches for a `[Ref Code: XX-000]` pattern using regex and returns the code string (e.g. `"RW-001"`). Returns `""` if no match is found or if input is not a string. |
| `normalize_text` | `(topic: str, content: str) → str` | Cleans both inputs and concatenates them as `"Topic. Content"` — producing the final text string that is fed to the embedding model. |

**Example:**

```python
from Utils.text_cleaning import normalize_text, extract_ref_code

content = "Employees can work remotely.  [Ref Code: RW-001]"
ref = extract_ref_code(content)        # "RW-001"
text = normalize_text("Remote Work", content)
# "Remote Work. Employees can work remotely. [Ref Code: RW-001]"
```

**Dependencies:** `re` (standard library only)

---

### `formatting.py` — Terminal Colours, Banners & Output Formatting

Handles all user-facing terminal output: ANSI colours, Unicode box-drawing, status messages, the startup banner, and formatted answer display.

#### Colour & Encoding Support

| Feature | Behaviour |
|---|---|
| **ANSI colour detection** | Auto-detects support on Windows Terminal, VS Code integrated terminal, `ANSICON`, and POSIX ttys. |
| **`NO_COLOR` compliance** | Respects the [`NO_COLOR`](https://no-color.org/) environment variable — all colours and formatting are disabled when set. |
| **Unicode fallback** | Each Unicode symbol and box-drawing character is tested at import time; if `stdout` cannot encode it, an ASCII equivalent is used instead. |
| **Windows UTF-8** | On Windows, `stdout` and `stderr` are reconfigured to UTF-8 encoding at import time for consistent output. |

#### Constants

**Formatting codes** (empty strings when colour is disabled):

| Constant | ANSI Code | Purpose |
|---|---|---|
| `RESET` | `\033[0m` | Reset all formatting |
| `BOLD` | `\033[1m` | Bold text |
| `DIM` | `\033[2m` | Dimmed text |
| `CYAN` | `\033[36m` | Cyan foreground |
| `GREEN` | `\033[32m` | Green foreground |
| `YELLOW` | `\033[33m` | Yellow foreground |
| `RED` | `\033[31m` | Red foreground |
| `MAGENTA` | `\033[35m` | Magenta foreground |
| `BLUE` | `\033[34m` | Blue foreground |
| `WHITE` | `\033[97m` | Bright white foreground |

**Symbols** (with ASCII fallbacks):

| Constant | Unicode | ASCII Fallback | Usage |
|---|---|---|---|
| `SYM_BULLET` | `•` | `*` | List bullets |
| `SYM_ARROW` | `▸` | `>` | Status indicators |
| `SYM_CHECK` | `✓` | `+` | Success messages |
| `SYM_WARN` | `⚠` | `!` | Warnings / errors |
| `SYM_SEARCH` | `🔍` | `?` | Question header |
| `SYM_BOOK` | `📖` | `#` | Snippets header |
| `SYM_BRAIN` | `🤖` | `*` | Answer header |

**Box-drawing characters** (with ASCII fallbacks):

| Constant | Unicode | ASCII | Description |
|---|---|---|---|
| `BOX_H` | `─` | `-` | Horizontal line |
| `BOX_V` | `│` | `\|` | Vertical line |
| `BOX_TL` | `╭` | `+` | Top-left corner |
| `BOX_TR` | `╮` | `+` | Top-right corner |
| `BOX_BL` | `╰` | `+` | Bottom-left corner |
| `BOX_BR` | `╯` | `+` | Bottom-right corner |
| `BOX_HH` | `═` | `=` | Heavy horizontal line |

**Layout:** `WIDTH = 72` (default terminal output width)

#### Public Functions

| Function | Signature | Description |
|---|---|---|
| `print_banner` | `() → None` | Prints a styled startup banner with the chatbot title, description, and quit instructions inside a Unicode box. |
| `print_separator` | `(label: str = '', style: str = 'light') → None` | Prints a horizontal rule. `style='light'` uses `─`, `style='heavy'` uses `═`. Optional centred label. |
| `print_status` | `(message: str, style: str = 'info') → None` | Prints a coloured status line with an icon. Styles: `info`, `success`, `warn`, `error`, `load`, `build`. |
| `print_prompt` | `() → str` | Displays the input prompt (`? Ask a policy question:`) and returns the user's trimmed input. Returns `""` on `EOFError`. |
| `print_question` | `(query: str) → None` | Prints the user's question in a highlighted block with a 🔍 header. |
| `print_snippets_header` | `() → None` | Prints the `📖 RETRIEVED SNIPPETS` section header. |
| `print_no_results` | `() → None` | Prints a yellow warning when no relevant snippets are found. |
| `print_answer_header` | `() → None` | Prints the `🤖 ANSWER` section header with a heavy separator. |
| `print_goodbye` | `() → None` | Prints a farewell message (`✓ Goodbye!`). |
| `format_context` | `(chunks: List[Dict]) → str` | Formats a list of retrieved chunks into a numbered, indented, word-wrapped string for terminal display. Each entry shows its index, ref code, topic, and wrapped content. |
| `format_response` | `(answer: str, chunks: List[Dict]) → str` | Formats the generated answer with word-wrapping and appends a `Sources:` line listing unique ref codes from the cited chunks. |

**Dependencies:** `os`, `sys`, `textwrap` (standard library only)
