import os
import sys
import textwrap
from typing import List, Dict


# ── Force UTF-8 on Windows to support box-drawing & special chars ─────
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ── ANSI color helpers ────────────────────────────────────────────────
_NO_COLOR = os.getenv('NO_COLOR')  # respect https://no-color.org/


def _supports_color() -> bool:
    if _NO_COLOR is not None:
        return False
    if sys.platform == 'win32':
        # Windows Terminal, VS Code, and modern terminals support ANSI
        return bool(os.getenv('WT_SESSION') or os.getenv('TERM_PROGRAM') or os.getenv('ANSICON'))
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


_COLOR = _supports_color()

# Colors
RESET   = '\033[0m'   if _COLOR else ''
BOLD    = '\033[1m'    if _COLOR else ''
DIM     = '\033[2m'    if _COLOR else ''
CYAN    = '\033[36m'   if _COLOR else ''
GREEN   = '\033[32m'   if _COLOR else ''
YELLOW  = '\033[33m'   if _COLOR else ''
RED     = '\033[31m'   if _COLOR else ''
MAGENTA = '\033[35m'   if _COLOR else ''
BLUE    = '\033[34m'   if _COLOR else ''
WHITE   = '\033[97m'   if _COLOR else ''


def _can_encode(char: str) -> bool:
    """Check if stdout can encode a character."""
    try:
        char.encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Symbols — with ASCII fallbacks for legacy terminals
SYM_BULLET = '*' if not _can_encode('\u2022') else '\u2022'   # •
SYM_ARROW  = '>' if not _can_encode('\u25b8') else '\u25b8'   # ▸
SYM_CHECK  = '+' if not _can_encode('\u2713') else '\u2713'   # ✓
SYM_WARN   = '!' if not _can_encode('\u26a0') else '\u26a0'   # ⚠
SYM_SEARCH = '?' if not _can_encode('\U0001f50d') else '\U0001f50d'  # 🔍
SYM_BOOK   = '#' if not _can_encode('\U0001f4d6') else '\U0001f4d6'  # 📖
SYM_BRAIN  = '*' if not _can_encode('\U0001f916') else '\U0001f916'  # 🤖

# Box-drawing — with ASCII fallbacks
BOX_H    = '-' if not _can_encode('\u2500') else '\u2500'   # ─
BOX_V    = '|' if not _can_encode('\u2502') else '\u2502'   # │
BOX_TL   = '+' if not _can_encode('\u256d') else '\u256d'   # ╭
BOX_TR   = '+' if not _can_encode('\u256e') else '\u256e'   # ╮
BOX_BL   = '+' if not _can_encode('\u2570') else '\u2570'   # ╰
BOX_BR   = '+' if not _can_encode('\u256f') else '\u256f'   # ╯
BOX_HH   = '=' if not _can_encode('\u2550') else '\u2550'   # ═

WIDTH = 72


# ── Primitives ────────────────────────────────────────────────────────

def _box_line(text: str, width: int = WIDTH) -> str:
    """Centre text in a bordered line."""
    inner = width - 4
    return f'{BOX_V} {text:^{inner}} {BOX_V}'


def _box_top(width: int = WIDTH) -> str:
    return BOX_TL + BOX_H * (width - 2) + BOX_TR


def _box_bottom(width: int = WIDTH) -> str:
    return BOX_BL + BOX_H * (width - 2) + BOX_BR


# ── Public API ────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print a styled startup banner."""
    print()
    print(f'{CYAN}{_box_top()}{RESET}')
    print(f'{CYAN}{_box_line("")}{RESET}')
    print(f'{CYAN}{_box_line(f"{BOLD}{WHITE}RAG Policy Chatbot{RESET}{CYAN}")}{RESET}')
    print(f'{CYAN}{_box_line(f"{DIM}Retrieval-augmented answers from company policy data{RESET}{CYAN}")}{RESET}')
    print(f'{CYAN}{_box_line("")}{RESET}')
    print(f'{CYAN}{_box_line(f"{DIM}Type {YELLOW}quit{RESET}{DIM} or {YELLOW}exit{RESET}{DIM} to stop{RESET}{CYAN}")}{RESET}')
    print(f'{CYAN}{_box_line("")}{RESET}')
    print(f'{CYAN}{_box_bottom()}{RESET}')
    print()


def print_separator(label: str = '', style: str = 'light') -> None:
    """Print a horizontal rule, optionally with a centred label."""
    char = BOX_H if style == 'light' else BOX_HH
    color = DIM if style == 'light' else CYAN
    if label:
        pad = max((WIDTH - len(label) - 4) // 2, 1)
        line = f'{char * pad} {label} {char * pad}'
        # handle odd widths
        diff = WIDTH - len(line)
        if diff > 0:
            line += char * diff
    else:
        line = char * WIDTH
    print(f'{color}{line}{RESET}')


def print_status(message: str, style: str = 'info') -> None:
    """Print a coloured status message (info, success, warn, error)."""
    styles = {
        'info':    (CYAN,    SYM_ARROW),
        'success': (GREEN,   SYM_CHECK),
        'warn':    (YELLOW,  SYM_WARN),
        'error':   (RED,     SYM_WARN),
        'load':    (BLUE,    SYM_ARROW),
        'build':   (MAGENTA, SYM_ARROW),
    }
    color, sym = styles.get(style, (CYAN, SYM_ARROW))
    print(f'  {color}{sym}{RESET} {message}')


def print_prompt() -> str:
    """Display the input prompt and return the user's query."""
    try:
        return input(f'\n  {GREEN}{BOLD}?{RESET} {BOLD}Ask a policy question:{RESET} ').strip()
    except EOFError:
        return ''


def print_question(query: str) -> None:
    """Print the user's question in a highlighted block."""
    print()
    print_separator(f'{SYM_SEARCH} YOUR QUESTION')
    print(f'  {BOLD}{WHITE}{query}{RESET}')
    print()


def print_snippets_header() -> None:
    print_separator(f'{SYM_BOOK} RETRIEVED SNIPPETS')


def print_no_results() -> None:
    print(f'\n  {YELLOW}{SYM_WARN}{RESET} {DIM}No relevant policy snippets found.{RESET}\n')


def print_answer_header() -> None:
    print()
    print_separator(f'{SYM_BRAIN} ANSWER', style='heavy')


def print_goodbye() -> None:
    print(f'\n  {CYAN}{SYM_CHECK}{RESET} {DIM}Goodbye!{RESET}\n')


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks for display with clean numbering."""
    if not chunks:
        return ''
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        ref_code = chunk.get('ref_code', '')
        topic = chunk.get('topic', '')
        content = chunk.get('content', '').strip()

        # Header: number + ref code + topic
        ref_part = f'{DIM}[{ref_code}]{RESET} ' if ref_code else ''
        header = f'  {CYAN}{BOLD}{i}.{RESET} {ref_part}{BOLD}{topic}{RESET}'
        lines.append(header)

        # Content (indented, wrapped)
        if content:
            wrapped = textwrap.fill(content, width=WIDTH - 8, initial_indent='     ', subsequent_indent='     ')
            lines.append(f'{DIM}{wrapped}{RESET}')
        lines.append('')  # blank line between snippets
    return '\n'.join(lines)


def format_response(answer: str, chunks: List[Dict]) -> str:
    """Format the final answer with optional source references."""
    if not answer:
        return f'  {DIM}No answer was generated.{RESET}'

    # Wrap the answer text
    wrapped = textwrap.fill(answer.strip(), width=WIDTH - 4, initial_indent='  ', subsequent_indent='  ')
    result = f'{WHITE}{wrapped}{RESET}'

    # Append sources
    sources = sorted({chunk.get('ref_code', '') for chunk in chunks if chunk.get('ref_code')})
    if sources:
        src_text = ', '.join(sources)
        result += f'\n\n  {DIM}Sources: {CYAN}{src_text}{RESET}'

    return result
