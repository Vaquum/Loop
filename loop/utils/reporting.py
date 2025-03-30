"""
Utility functions for standardized report formatting.
"""

def format_report_header(title: str, width: int = 50) -> str:
    """
    Format a standardized report header with consistent styling.
    
    Parameters:
        title: The title of the report section
        width: Width of the separator line (default: 50)
        
    Returns:
        Formatted header string
    """
    separator = "=" * width
    return f"\n{separator}\n{title}\n{separator}"

def format_report_section(title: str, width: int = 50) -> str:
    """
    Format a standardized report section with consistent styling.
    
    Parameters:
        title: The title of the report section
        width: Width of the separator line (default: 50)
        
    Returns:
        Formatted section string
    """
    separator = "-" * width
    return f"\n{separator}\n{title}\n{separator}"

def format_report_footer(width: int = 50) -> str:
    """
    Format a standardized report footer with consistent styling.
    
    Parameters:
        width: Width of the separator line (default: 50)
        
    Returns:
        Formatted footer string
    """
    return "=" * width 