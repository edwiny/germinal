# Purpose: Tools for accessing large content that exceeds context limits.
#          Stores piped stdin content and provides tools to access portions
#          incrementally, allowing agents to work with large files without
#          fitting everything in the context window at once.
#
# Relationships: Content is stored globally during main execution and
#               accessed by tools during agent invocation.

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, model_to_json_schema

# Global storage for large content that exceeds context limits
# This is set in __main__.py when content is too large to send directly
_large_content_store: Optional[str] = None


def set_large_content(content: str) -> None:
    """Store large content for incremental access by tools."""
    global _large_content_store
    _large_content_store = content


def get_large_content() -> Optional[str]:
    """Get the stored large content."""
    return _large_content_store


def has_large_content() -> bool:
    """Check if large content is available."""
    return _large_content_store is not None


# ---------------------------------------------------------------------------
# read_content_range
# ---------------------------------------------------------------------------

class ReadContentRangeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(
        description="Starting line number (1-indexed).",
        ge=1
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (inclusive). If not provided, reads to end of content.",
        ge=1
    )
    max_chars: Optional[int] = Field(
        default=10000,
        description="Maximum characters to return. Truncates if exceeded.",
        ge=1,
        le=50000
    )


class ReadContentRangeResult(BaseModel):
    content: str = Field(description="The requested content range.")
    start_line: int = Field(description="Actual starting line number.")
    end_line: int = Field(description="Actual ending line number.")
    total_lines: int = Field(description="Total lines in the content.")
    truncated: bool = Field(description="True if content was truncated due to max_chars limit.")


def make_read_content_range_tool() -> Tool:
    """Return a tool for reading ranges of large content."""

    def execute(params: dict) -> dict:
        if not has_large_content():
            return {"error": "No large content available. This tool only works with piped input that exceeded context limits."}

        content = get_large_content()
        lines = content.splitlines()
        total_lines = len(lines)

        start_line = params["start_line"] - 1  # Convert to 0-indexed
        end_line = params.get("end_line")
        max_chars = params.get("max_chars", 10000)

        if start_line >= total_lines:
            return {"error": f"Start line {params['start_line']} exceeds total lines ({total_lines})"}

        if end_line is None:
            end_line = total_lines
        else:
            end_line = min(end_line, total_lines)

        # Extract the requested line range
        selected_lines = lines[start_line:end_line]
        content_range = '\n'.join(selected_lines)

        # Apply character limit
        truncated = False
        if len(content_range) > max_chars:
            content_range = content_range[:max_chars]
            truncated = True

        return ReadContentRangeResult(
            content=content_range,
            start_line=start_line + 1,  # Convert back to 1-indexed
            end_line=min(end_line, len(selected_lines) + start_line),
            total_lines=total_lines,
            truncated=truncated
        ).model_dump()

    return Tool(
        name="read_content_range",
        description=(
            "Read a specific range of lines from large piped input content. "
            "Use this when the input data was too large to fit in the initial context. "
            "Lines are 1-indexed. Content is truncated if it exceeds max_chars."
        ),
        parameters_schema=model_to_json_schema(ReadContentRangeParams),
        risk_level="low",
        _execute=execute,
        params_model=ReadContentRangeParams,
    )


# ---------------------------------------------------------------------------
# search_content
# ---------------------------------------------------------------------------

class SearchContentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Text pattern to search for (case-sensitive).")
    max_results: Optional[int] = Field(
        default=10,
        description="Maximum number of matches to return.",
        ge=1,
        le=50
    )
    context_lines: Optional[int] = Field(
        default=2,
        description="Number of lines of context around each match.",
        ge=0,
        le=10
    )


class ContentMatch(BaseModel):
    line_number: int = Field(description="Line number where match was found (1-indexed).")
    content: str = Field(description="The matching line with context.")


class SearchContentResult(BaseModel):
    matches: list[dict] = Field(description="List of matches found.")
    total_matches: int = Field(description="Total number of matches found.")
    truncated: bool = Field(description="True if results were truncated due to max_results limit.")


def make_search_content_tool() -> Tool:
    """Return a tool for searching within large content."""

    def execute(params: dict) -> dict:
        if not has_large_content():
            return {"error": "No large content available. This tool only works with piped input that exceeded context limits."}

        content = get_large_content()
        lines = content.splitlines()
        pattern = params["pattern"]
        max_results = params.get("max_results", 10)
        context_lines = params.get("context_lines", 2)

        matches = []
        for i, line in enumerate(lines, 1):  # 1-indexed line numbers
            if pattern in line:
                # Build context around the match
                start_ctx = max(0, i - context_lines - 1)
                end_ctx = min(len(lines), i + context_lines)

                context_lines_list = []
                for ctx_idx in range(start_ctx, end_ctx):
                    marker = ">>> " if ctx_idx + 1 == i else "    "
                    context_lines_list.append(f"{marker}{ctx_idx + 1:4d}: {lines[ctx_idx]}")

                match_content = '\n'.join(context_lines_list)
                matches.append(ContentMatch(line_number=i, content=match_content).model_dump())

                if len(matches) >= max_results:
                    break

        truncated = len(matches) >= max_results

        return SearchContentResult(
            matches=matches,
            total_matches=len(matches),
            truncated=truncated
        ).model_dump()

    return Tool(
        name="search_content",
        description=(
            "Search for text patterns within large piped input content. "
            "Returns matches with surrounding context lines. "
            "Use this to find specific information in large files."
        ),
        parameters_schema=model_to_json_schema(SearchContentParams),
        risk_level="low",
        _execute=execute,
        params_model=SearchContentParams,
    )


# ---------------------------------------------------------------------------
# get_content_info
# ---------------------------------------------------------------------------

class GetContentInfoResult(BaseModel):
    available: bool = Field(description="True if large content is available.")
    total_lines: Optional[int] = Field(description="Total number of lines in the content.")
    total_chars: Optional[int] = Field(description="Total number of characters in the content.")
    estimated_tokens: Optional[int] = Field(description="Estimated token count (~4 chars per token).")


def make_get_content_info_tool() -> Tool:
    """Return a tool for getting information about stored large content."""

    def execute(params: dict) -> dict:
        if not has_large_content():
            return GetContentInfoResult(available=False).model_dump()

        content = get_large_content()
        lines = content.splitlines()
        total_lines = len(lines)
        total_chars = len(content)
        estimated_tokens = total_chars // 4  # Rough estimate

        return GetContentInfoResult(
            available=True,
            total_lines=total_lines,
            total_chars=total_chars,
            estimated_tokens=estimated_tokens
        ).model_dump()

    return Tool(
        name="get_content_info",
        description=(
            "Get information about large piped input content. "
            "Returns line count, character count, and token estimates."
        ),
        parameters_schema={"type": "object", "properties": {}},  # No parameters
        risk_level="low",
        _execute=execute,
        params_model=None,
    )