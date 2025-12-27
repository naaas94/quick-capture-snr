#!/usr/bin/env python3
"""
QuickCapture — Main Entry Point

Enhanced symbolic ingestion layer for structured note capture with downstream
compatibility for semantic systems such as the Semantic Note Router (SNR).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Import our modules
from scripts.parse_input import parse_note_input
from scripts.validate_note import validate_note
from scripts.models import ParsedNote, create_note_from_parsed
from scripts.storage_engine import StorageEngine
# from observability.metrics_collector import QuickCaptureMetrics

console = Console()


def main():
    """Main entry point for QuickCapture CLI."""
    parser = argparse.ArgumentParser(
        description="QuickCapture — Enhanced Symbolic Ingestion Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quick_add "python, coding: Implemented new feature for data processing : This will help with the ML pipeline"
  quick_add "meeting, project: Discussed Q4 roadmap with team : Need to follow up on budget approval"
  quick_add "idea, ml: Consider using transformer models for text classification : Research BERT vs RoBERTa"
        """
    )
    
    parser.add_argument(
        "input_text",
        help="Note input in format: 'tag1, tag2: note body : optional comment'"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed processing information"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without storing (for testing)"
    )
    
    args = parser.parse_args()
    
    # Display welcome message
    console.print(Panel(
        Text("QuickCapture — Enhanced Symbolic Ingestion Layer", style="bold blue"),
        subtitle="SNR Ecosystem / Semantic Note Infrastructure",
        border_style="blue"
    ))
    
    try:
        # TODO: Implement the full pipeline in subsequent stages
        console.print(f"[green]Processing input:[/green] {args.input_text}")
        
        if args.verbose:
            console.print("[yellow]Verbose mode enabled[/yellow]")
        
        if args.dry_run:
            console.print("[yellow]Dry run mode - no storage operations[/yellow]")
        
        # Full pipeline implementation
        # 1. Parse input
        if args.verbose:
            console.print("[blue]Parsing input...[/blue]")
        
        parsed = parse_note_input(args.input_text)
        
        if args.verbose:
            console.print(f"[green]✓ Parsed successfully[/green]")
            console.print(f"  Tags: {parsed['tags']}")
            console.print(f"  Note: {parsed['note']}")
            console.print(f"  Comment: {parsed['comment']}")
            console.print(f"  Semantic Density: {parsed['semantic_density']}")
            console.print(f"  Content Type: {parsed['content_type'].value}")
            console.print(f"  Confidence: {parsed['confidence_score']}")
        
        # 2. Validate note
        if args.verbose:
            console.print("[blue]Validating note...[/blue]")
        
        validation = validate_note(parsed)
        
        if args.verbose:
            console.print(f"[green]✓ Validation completed[/green]")
            console.print(f"  Valid: {validation['valid']}")
            console.print(f"  Tag Quality: {validation['tag_quality_score']}")
            console.print(f"  Semantic Coherence: {validation['semantic_coherence_score']}")
            console.print(f"  Overall Confidence: {validation['overall_confidence']}")
            
            if validation['issues']:
                console.print("[yellow]Issues found:[/yellow]")
                for issue in validation['issues']:
                    console.print(f"  - {issue['message']}")
        
        # 3. Create ParsedNote object
        if args.verbose:
            console.print("[blue]Creating note object...[/blue]")
        
        note = create_note_from_parsed(parsed, validation)
        
        if args.verbose:
            console.print(f"[green]✓ Note object created[/green]")
            console.print(f"  Note ID: {note.note_id}")
            console.print(f"  Timestamp: {note.timestamp}")
        
        # 4. Store note (if not dry run)
        if not args.dry_run:
            if args.verbose:
                console.print("[blue]Storing note...[/blue]")
            
            storage = StorageEngine()
            success = storage.store_note(note)
            
            if success:
                console.print("[green]✓ Note stored successfully[/green]")
                
                # Show database stats
                if args.verbose:
                    stats = storage.get_database_stats()
                    console.print(f"[blue]Database stats:[/blue] {stats['total_notes']} total notes, {stats['total_tags']} tags")
            else:
                console.print("[red]✗ Failed to store note[/red]")
        else:
            console.print("[yellow]Dry run - note not stored[/yellow]")
        
        console.print("[green]✓ Processing completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main() 