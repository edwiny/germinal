import argparse
import asyncio
import os
import sys
from importlib.resources import files
from pathlib import Path

from .core.config import config
from .main_loop import main as _async_main
from .main_interactive import run_interactive
from .tools.content_access import set_large_content

_CONFIG_PATH = Path.home() / ".config" / "germinal" / "config.yaml"


def _init_config() -> None:
    """
    Copy the bundled default config to ~/.config/germinal/config.yaml if it
    does not already exist.

    Uses importlib.resources so this works whether germinal is installed from
    a wheel or run directly from source via 'uv run germ'.
    """
    if _CONFIG_PATH.exists():
        return
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    default = files("orchestrator").joinpath("config.yaml").read_bytes()
    _CONFIG_PATH.write_bytes(default)
    print(f"Created default config: {_CONFIG_PATH}")
    print("Edit it before running germ again.")


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token for most English text.
    This is a conservative estimate - actual tokenization may vary.
    """
    return len(text) // 4


def _check_stdin_size() -> tuple[int, bool]:
    """
    Check the size of data available on stdin without reading it.
    Returns (size_in_bytes, is_data_available).
    """
    if sys.stdin.isatty():
        return 0, False

    try:
        # Use os.fstat to check size if stdin is a regular file/pipe
        # This works for pipes and redirected files
        stat = os.fstat(sys.stdin.fileno())
        size = stat.st_size
        return size, size > 0
    except (OSError, AttributeError):
        # Fallback: try to peek at stdin
        try:
            import select
            if hasattr(select, 'select') and sys.stdin.fileno() >= 0:
                # Check if data is available to read
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    # Peek at first byte to confirm data is available
                    data = sys.stdin.read(1)
                    if data:
                        # Put the byte back
                        sys.stdin.seek(0) if hasattr(sys.stdin, 'seek') else None
                        return 1, True  # At least 1 byte available
        except (ImportError, OSError, AttributeError):
            pass
        return 0, False


def _read_stdin_with_limits() -> str | None:
    """
    Read stdin content with size validation and token limits.
    Returns the content as a string, or None if no data available.
    Exits with error if content exceeds configured limits.
    """
    if sys.stdin.isatty():
        return None

    # Check size before reading
    size_bytes, has_data = _check_stdin_size()
    if not has_data:
        return None

    input_config = config.get("input", {})
    max_size_mb = input_config.get("max_file_size_mb", 100)
    max_tokens = input_config.get("max_tokens_estimate", 200000)
    large_threshold_mb = input_config.get("large_file_threshold_mb", 10)

    max_size_bytes = max_size_mb * 1024 * 1024

    if size_bytes > max_size_bytes:
        print(f"error: Input data size ({size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)", file=sys.stderr)
        print("Consider using a file path instead of piping, or reduce the input size.", file=sys.stderr)
        sys.exit(1)

    if size_bytes > large_threshold_mb * 1024 * 1024:
        print(f"warning: Large input detected ({size_bytes / (1024*1024):.1f} MB). This may take time to process.", file=sys.stderr)

    try:
        content = sys.stdin.read()
    except Exception as e:
        print(f"error: Failed to read stdin: {e}", file=sys.stderr)
        sys.exit(1)

    # Check token estimate
    estimated_tokens = _estimate_tokens(content)
    if estimated_tokens > max_tokens:
        print(f"error: Estimated input size ({estimated_tokens:,} tokens) exceeds maximum allowed ({max_tokens:,} tokens)", file=sys.stderr)
        print("The model context window may be exceeded. Consider summarizing the input or using a smaller file.", file=sys.stderr)
        sys.exit(1)

    if estimated_tokens > max_tokens * 0.8:  # Warning at 80% of limit
        print(f"warning: Input is estimated at {estimated_tokens:,} tokens, approaching context limit.", file=sys.stderr)

    return content


def main() -> None:
    _init_config()

    # Config is loaded automatically when imported
    # No need to call load_config() as it's done in Config.__init__

    parser = argparse.ArgumentParser(prog="germ", description="Germinal agent")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (network adapter + event loop)",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="One-shot prompt; if omitted, enter interactive REPL",
    )
    args = parser.parse_args()

    if args.daemon:
        asyncio.run(_async_main())
    else:
        # Check for piped stdin content with size validation
        stdin_content = _read_stdin_with_limits()

        # Determine how to handle the content
        combined_prompt = args.prompt
        if stdin_content:
            input_config = config.get("input", {})
            max_tokens = input_config.get("max_tokens_estimate", 200000)
            estimated_tokens = _estimate_tokens(stdin_content)

            if estimated_tokens > max_tokens * 0.8:  # Content is approaching limit
                # Store content for incremental access and provide summary
                set_large_content(stdin_content)

                # Create a summary prompt that tells the agent about the large content
                lines = stdin_content.splitlines()
                total_lines = len(lines)
                total_chars = len(stdin_content)

                content_summary = (
                    f"Large input data has been received ({total_lines:,} lines, {total_chars:,} characters, "
                    f"~{estimated_tokens:,} tokens). The full content is too large to include in the initial "
                    f"context window. Use the available content access tools to examine specific portions "
                    f"of the data as needed."
                )

                if args.prompt:
                    combined_prompt = f"{args.prompt}\n\n{content_summary}"
                else:
                    combined_prompt = content_summary
            else:
                # Content fits in context, include it normally
                if args.prompt:
                    combined_prompt = f"{args.prompt}\n\nInput data:\n{stdin_content}"
                else:
                    combined_prompt = stdin_content

        asyncio.run(run_interactive(prompt=combined_prompt))


if __name__ == "__main__":
    main()
