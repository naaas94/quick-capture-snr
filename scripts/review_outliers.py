#!/usr/bin/env python3
"""
QuickCapture Outlier Review Interface

Intelligent inspection and correction of malformed notes with semantic assistance.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from scripts.storage_engine import StorageEngine
from scripts.models import ParsedNote
from scripts.parse_input import parse_note_input
from scripts.validate_note import validate_note
from scripts.tag_intelligence import TagIntelligence
from observability.metrics_collector import record_error

console = Console()


class OutlierReviewer:
    """Intelligent outlier review and correction system."""
    
    def __init__(self):
        self.storage = StorageEngine()
        self.tag_intelligence = TagIntelligence(self.storage)
        self.correction_history = []
    
    def get_outliers(self, tag: Optional[str] = None, limit: int = 50, 
                    min_confidence: float = 0.5, include_issues: bool = True) -> List[ParsedNote]:
        """
        Get notes that are potential outliers based on various criteria.
        
        Args:
            tag: Filter by specific tag
            limit: Maximum number of notes to return
            min_confidence: Minimum confidence threshold
            include_issues: Include notes with validation issues
            
        Returns:
            List of outlier notes
        """
        outliers = []
        
        if tag:
            # Get notes by tag
            notes = self.storage.retrieve_notes_by_tag(tag, limit * 2)
        else:
            # Get notes with low confidence or issues
            notes = self.storage.get_notes_by_confidence_range(0.0, min_confidence, limit * 2)
        
        # Filter for outliers
        for note in notes:
            is_outlier = False
            
            # Low confidence
            if note.confidence_score < min_confidence:
                is_outlier = True
            
            # Validation issues
            if include_issues and note.issues:
                is_outlier = True
            
            # Low semantic density
            if note.semantic_density < 0.3:
                is_outlier = True
            
            # Low tag quality
            if note.tag_quality_score < 0.6:
                is_outlier = True
            
            if is_outlier:
                outliers.append(note)
                if len(outliers) >= limit:
                    break
        
        return outliers
    
    def display_note(self, note: ParsedNote, show_suggestions: bool = True):
        """Display a note with detailed information and suggestions."""
        console.print(Panel(
            f"[bold blue]Note ID:[/bold blue] {note.note_id}\n"
            f"[bold blue]Timestamp:[/bold blue] {note.timestamp}\n"
            f"[bold blue]Tags:[/bold blue] {', '.join(note.tags)}\n"
            f"[bold blue]Note:[/bold blue] {note.note}\n"
            f"[bold blue]Comment:[/bold blue] {note.comment or 'None'}\n"
            f"[bold blue]Content Type:[/bold blue] {note.content_type.value}\n"
            f"[bold blue]Confidence:[/bold blue] {note.confidence_score:.2f}\n"
            f"[bold blue]Semantic Density:[/bold blue] {note.semantic_density:.2f}\n"
            f"[bold blue]Tag Quality:[/bold blue] {note.tag_quality_score:.2f}\n"
            f"[bold blue]Valid:[/bold blue] {'✓' if note.valid else '✗'}\n"
            f"[bold blue]Issues:[/bold blue] {len(note.issues)}",
            title="Note Details",
            border_style="blue"
        ))
        
        if note.issues:
            console.print(Panel(
                "\n".join([f"• {issue}" for issue in note.issues]),
                title="Validation Issues",
                border_style="red"
            ))
        
        if show_suggestions:
            self._show_suggestions(note)
    
    def _show_suggestions(self, note: ParsedNote):
        """Show intelligent suggestions for improving the note."""
        suggestions = []
        
        # Tag suggestions
        if note.tag_quality_score < 0.8:
            tag_suggestions = self.tag_intelligence.suggest_tags(note.note, note.tags)
            if tag_suggestions:
                suggestions.append(f"Tag suggestions: {', '.join([s.tag for s in tag_suggestions[:3]])}")
        
        # Content suggestions
        if note.semantic_density < 0.4:
            suggestions.append("Consider adding more specific content to improve semantic density")
        
        if len(note.note) < 20:
            suggestions.append("Note body is very short - consider expanding with more details")
        
        if not note.comment:
            suggestions.append("Consider adding a comment for additional context")
        
        # Format suggestions
        if not note.tags:
            suggestions.append("Add relevant tags to improve categorization")
        
        if suggestions:
            console.print(Panel(
                "\n".join([f"💡 {suggestion}" for suggestion in suggestions]),
                title="Improvement Suggestions",
                border_style="yellow"
            ))
    
    def edit_note(self, note: ParsedNote) -> Optional[ParsedNote]:
        """Interactive note editing with semantic assistance."""
        console.print("\n[bold green]Editing Note[/bold green]")
        console.print("Press Enter to keep current value, or type new value:")
        
        # Edit tags
        current_tags = ', '.join(note.tags)
        new_tags_input = Prompt.ask(
            f"Tags (comma-separated)",
            default=current_tags
        )
        
        if new_tags_input != current_tags:
            new_tags = [tag.strip() for tag in new_tags_input.split(',') if tag.strip()]
        else:
            new_tags = note.tags
        
        # Edit note body
        new_note_body = Prompt.ask(
            f"Note body",
            default=note.note
        )
        
        # Edit comment
        new_comment = Prompt.ask(
            f"Comment (optional)",
            default=note.comment or ""
        )
        
        if not new_comment.strip():
            new_comment = None
        
        # Reconstruct the input text
        new_input_text = f"{', '.join(new_tags)}: {new_note_body}"
        if new_comment:
            new_input_text += f" : {new_comment}"
        
        try:
            # Re-parse the edited input
            parsed = parse_note_input(new_input_text)
            validation = validate_note(parsed)
            
            # Create new note with updated data
            corrected_note = ParsedNote(
                note_id=note.note_id,  # Keep same ID
                tags=new_tags,
                note=new_note_body,
                comment=new_comment,
                timestamp=note.timestamp,
                valid=validation['valid'],
                issues=validation.get('issues', []),
                origin=note.origin,
                version=note.version + 1,  # Increment version
                raw_text=new_input_text,
                semantic_density=parsed['semantic_density'],
                tag_quality_score=validation['tag_quality_score'],
                content_type=parsed['content_type'],
                confidence_score=parsed['confidence_score']
            )
            
            # Show comparison
            self._show_correction_comparison(note, corrected_note)
            
            if Confirm.ask("Apply these corrections?"):
                # Store the corrected note
                success = self.storage.store_note(corrected_note)
                if success:
                    console.print("[green]✓ Note corrected and stored[/green]")
                    
                    # Record correction
                    self.correction_history.append({
                        'original_note_id': note.note_id,
                        'correction_timestamp': datetime.now().isoformat(),
                        'changes': {
                            'tags_changed': note.tags != new_tags,
                            'body_changed': note.note != new_note_body,
                            'comment_changed': note.comment != new_comment
                        },
                        'improvements': {
                            'confidence_improvement': corrected_note.confidence_score - note.confidence_score,
                            'semantic_density_improvement': corrected_note.semantic_density - note.semantic_density,
                            'tag_quality_improvement': corrected_note.tag_quality_score - note.tag_quality_score,
                            'validation_improvement': corrected_note.valid and not note.valid
                        }
                    })
                    
                    return corrected_note
                else:
                    console.print("[red]✗ Failed to store corrected note[/red]")
                    record_error("storage_error", "Failed to store corrected note")
            else:
                console.print("[yellow]Corrections cancelled[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error during correction: {e}[/red]")
            record_error("correction_error", str(e))
        
        return None
    
    def _show_correction_comparison(self, original: ParsedNote, corrected: ParsedNote):
        """Show comparison between original and corrected note."""
        table = Table(title="Correction Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Original", style="red")
        table.add_column("Corrected", style="green")
        table.add_column("Improvement", style="yellow")
        
        # Add comparison rows
        table.add_row(
            "Confidence",
            f"{original.confidence_score:.2f}",
            f"{corrected.confidence_score:.2f}",
            f"{corrected.confidence_score - original.confidence_score:+.2f}"
        )
        
        table.add_row(
            "Semantic Density",
            f"{original.semantic_density:.2f}",
            f"{corrected.semantic_density:.2f}",
            f"{corrected.semantic_density - original.semantic_density:+.2f}"
        )
        
        table.add_row(
            "Tag Quality",
            f"{original.tag_quality_score:.2f}",
            f"{corrected.tag_quality_score:.2f}",
            f"{corrected.tag_quality_score - original.tag_quality_score:+.2f}"
        )
        
        table.add_row(
            "Valid",
            "✓" if original.valid else "✗",
            "✓" if corrected.valid else "✗",
            "Improved" if corrected.valid and not original.valid else "Same"
        )
        
        table.add_row(
            "Issues",
            str(len(original.issues)),
            str(len(corrected.issues)),
            f"{len(original.issues) - len(corrected.issues):+d}"
        )
        
        console.print(table)
    
    def auto_fix_note(self, note: ParsedNote) -> Optional[ParsedNote]:
        """Attempt automatic correction of common issues."""
        console.print("[yellow]Attempting automatic correction...[/yellow]")
        
        try:
            # Common fixes
            fixed_tags = self._auto_fix_tags(note.tags)
            fixed_body = self._auto_fix_body(note.note)
            fixed_comment = note.comment
            
            # Reconstruct input
            fixed_input = f"{', '.join(fixed_tags)}: {fixed_body}"
            if fixed_comment:
                fixed_input += f" : {fixed_comment}"
            
            # Re-parse
            parsed = parse_note_input(fixed_input)
            validation = validate_note(parsed)
            
            # Create corrected note
            corrected_note = ParsedNote(
                note_id=note.note_id,
                tags=fixed_tags,
                note=fixed_body,
                comment=fixed_comment,
                timestamp=note.timestamp,
                valid=validation['valid'],
                issues=validation.get('issues', []),
                origin=note.origin,
                version=note.version + 1,
                raw_text=fixed_input,
                semantic_density=parsed['semantic_density'],
                tag_quality_score=validation['tag_quality_score'],
                content_type=parsed['content_type'],
                confidence_score=parsed['confidence_score']
            )
            
            # Check if auto-fix improved the note
            improvements = (
                corrected_note.confidence_score > note.confidence_score or
                corrected_note.semantic_density > note.semantic_density or
                corrected_note.tag_quality_score > note.tag_quality_score or
                (corrected_note.valid and not note.valid)
            )
            
            if improvements:
                self._show_correction_comparison(note, corrected_note)
                if Confirm.ask("Apply automatic corrections?"):
                    success = self.storage.store_note(corrected_note)
                    if success:
                        console.print("[green]✓ Auto-correction applied[/green]")
                        return corrected_note
                    else:
                        console.print("[red]✗ Failed to store auto-corrected note[/red]")
            else:
                console.print("[yellow]No improvements found with auto-correction[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Auto-correction failed: {e}[/red]")
            record_error("auto_correction_error", str(e))
        
        return None
    
    def _auto_fix_tags(self, tags: List[str]) -> List[str]:
        """Automatically fix common tag issues."""
        fixed_tags = []
        
        for tag in tags:
            # Normalize tag
            fixed_tag = tag.strip().lower()
            fixed_tag = fixed_tag.replace(' ', '_')
            fixed_tag = ''.join(c for c in fixed_tag if c.isalnum() or c in '_-')
            
            if fixed_tag and len(fixed_tag) >= 2:
                fixed_tags.append(fixed_tag)
        
        return fixed_tags
    
    def _auto_fix_body(self, body: str) -> str:
        """Automatically fix common body issues."""
        # Remove excessive whitespace
        fixed_body = ' '.join(body.split())
        
        # Capitalize first letter
        if fixed_body and fixed_body[0].islower():
            fixed_body = fixed_body[0].upper() + fixed_body[1:]
        
        # Add period if missing
        if fixed_body and not fixed_body.endswith(('.', '!', '?')):
            fixed_body += '.'
        
        return fixed_body
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about corrections made."""
        if not self.correction_history:
            return {"total_corrections": 0}
        
        total_corrections = len(self.correction_history)
        confidence_improvements = [c['improvements']['confidence_improvement'] for c in self.correction_history]
        semantic_improvements = [c['improvements']['semantic_density_improvement'] for c in self.correction_history]
        tag_improvements = [c['improvements']['tag_quality_improvement'] for c in self.correction_history]
        
        return {
            "total_corrections": total_corrections,
            "avg_confidence_improvement": sum(confidence_improvements) / len(confidence_improvements),
            "avg_semantic_improvement": sum(semantic_improvements) / len(semantic_improvements),
            "avg_tag_quality_improvement": sum(tag_improvements) / len(tag_improvements),
            "validation_improvements": sum(1 for c in self.correction_history if c['improvements']['validation_improvement']),
            "correction_rate": total_corrections / max(1, total_corrections)
        }


def main():
    """Main entry point for outlier review."""
    parser = argparse.ArgumentParser(
        description="QuickCapture Outlier Review - Intelligent note correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  review_outliers --tag python --limit 10
  review_outliers --edit --semantic
  review_outliers --auto-fix --limit 5
        """
    )
    
    parser.add_argument(
        "--tag",
        help="Filter outliers by specific tag"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum number of outliers to display (default: 20)"
    )
    
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Enable interactive editing mode"
    )
    
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic suggestions"
    )
    
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Attempt automatic corrections"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for outliers (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Initialize reviewer
    reviewer = OutlierReviewer()
    
    # Display welcome message
    console.print(Panel(
        Text("QuickCapture Outlier Review", style="bold blue"),
        subtitle="Intelligent Note Correction System",
        border_style="blue"
    ))
    
    try:
        # Get outliers
        outliers = reviewer.get_outliers(
            tag=args.tag,
            limit=args.limit,
            min_confidence=args.min_confidence
        )
        
        if not outliers:
            console.print("[green]No outliers found![/green]")
            return
        
        console.print(f"[blue]Found {len(outliers)} potential outliers[/blue]")
        
        # Display outliers
        for i, note in enumerate(outliers, 1):
            console.print(f"\n[bold cyan]Outlier {i}/{len(outliers)}[/bold cyan]")
            reviewer.display_note(note, show_suggestions=args.semantic)
            
            if args.edit:
                if Confirm.ask("Edit this note?"):
                    corrected = reviewer.edit_note(note)
                    if corrected:
                        note = corrected  # Update for next iteration
            
            elif args.auto_fix:
                if Confirm.ask("Attempt auto-correction?"):
                    corrected = reviewer.auto_fix_note(note)
                    if corrected:
                        note = corrected  # Update for next iteration
            
            if i < len(outliers):
                if not Confirm.ask("Continue to next outlier?"):
                    break
        
        # Show correction statistics
        if reviewer.correction_history:
            stats = reviewer.get_correction_statistics()
            console.print(Panel(
                f"Total corrections: {stats['total_corrections']}\n"
                f"Average confidence improvement: {stats['avg_confidence_improvement']:.3f}\n"
                f"Average semantic improvement: {stats['avg_semantic_improvement']:.3f}\n"
                f"Validation improvements: {stats['validation_improvements']}",
                title="Correction Statistics",
                border_style="green"
            ))
        
        console.print("[green]✓ Outlier review completed[/green]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        record_error("outlier_review_error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main() 