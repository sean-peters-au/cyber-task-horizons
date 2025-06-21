"""
Task analysis for identifying performance anomalies and unexpected difficulty patterns.

Creates visual tools for scanning task performance to identify potential issues with human time estimates.
"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .transform import transform_benchmark_results
from .. import config

logger = logging.getLogger(__name__)




def create_task_overview_table(
    output_dir: Optional[Path] = None,
    dataset_filter: Optional[str] = None
) -> Path:
    """
    Create a comprehensive HTML table showing all tasks with solve rates and model performance.
    
    Designed for visual scanning to identify tasks with unexpected difficulty patterns.
    
    Args:
        output_dir: Directory to save table (default: config.RESULTS_DIR / "plots")
        dataset_filter: Optional dataset name to filter results
        
    Returns:
        Path to generated HTML table file
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "plots"
    
    task_analysis_dir = output_dir / "task_analysis"
    task_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating task overview table...")
    
    # Get METR data
    metr_data_dir = output_dir / "metr_data"
    metr_data_dir.mkdir(exist_ok=True)
    files = transform_benchmark_results(metr_data_dir)
    
    # Load runs and apply dataset filter
    runs_df = pd.read_json(files['runs_file'], lines=True)
    if dataset_filter:
        runs_df = runs_df[runs_df['task_source'] == dataset_filter]
        logger.info(f"Filtered to {len(runs_df)} runs from dataset: {dataset_filter}")
    
    if runs_df.empty:
        logger.warning("No runs to analyze after filtering")
        output_file = task_analysis_dir / "task_overview.html"
        with open(output_file, 'w') as f:
            f.write("<html><body><h1>No data available</h1></body></html>")
        return output_file
    
    # Group by task to calculate solve rates and successful models
    task_summary = []
    
    for task_id, task_group in runs_df.groupby('task_id'):
        human_minutes = task_group['human_minutes'].iloc[0]
        task_source = task_group['task_source'].iloc[0]
        
        total_attempts = len(task_group)
        successful_attempts = task_group['score_binarized'].sum()
        solve_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
        
        # Get list of models that succeeded
        successful_models = task_group[task_group['score_binarized'] == 1]['alias'].unique().tolist()
        successful_models.sort()  # Consistent ordering
        
        task_summary.append({
            'task_id': task_id,
            'dataset': task_source,
            'human_minutes': human_minutes,
            'solve_rate': solve_rate,
            'solve_percentage': f"{solve_rate*100:.1f}%",
            'successful_models': successful_models,
            'models_list': ', '.join(successful_models) if successful_models else 'None',
            'total_attempts': total_attempts,
            'successful_attempts': int(successful_attempts)
        })
    
    # Sort by human time by default (makes anomalies easier to spot)
    task_summary.sort(key=lambda x: x['human_minutes'])
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Performance Overview{f' - {dataset_filter}' if dataset_filter else ''}</title>
    <style>
        body {{
            background-color: #1a1a1a;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .header {{
            max-width: 1400px;
            margin: 0 auto 30px auto;
            text-align: center;
        }}
        
        h1 {{
            color: #ffffff;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .subtitle {{
            color: #888;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .search-container {{
            margin-bottom: 20px;
        }}
        
        #searchInput {{
            background-color: #2d2d2d;
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 12px 16px;
            width: 300px;
            border-radius: 6px;
            font-size: 14px;
        }}
        
        #searchInput:focus {{
            outline: none;
            border-color: #0066cc;
            box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.3);
        }}
        
        .table-container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #2d2d2d;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        th {{
            background-color: #333;
            color: #fff;
            padding: 16px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #555;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        th:hover {{
            background-color: #404040;
        }}
        
        th .sort-arrow {{
            margin-left: 8px;
            opacity: 0.5;
        }}
        
        th.sort-asc .sort-arrow:after {{
            content: ' ↑';
            opacity: 1;
        }}
        
        th.sort-desc .sort-arrow:after {{
            content: ' ↓';
            opacity: 1;
        }}
        
        tr:nth-child(even) {{
            background-color: #252525;
        }}
        
        tr:nth-child(odd) {{
            background-color: #2d2d2d;
        }}
        
        tr:hover {{
            background-color: #3a3a3a;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #444;
            vertical-align: top;
        }}
        
        .task-id {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-size: 13px;
            color: #66d9ef;
            max-width: 200px;
            word-break: break-word;
        }}
        
        .dataset {{
            color: #a6e22e;
            font-weight: 500;
        }}
        
        .human-time {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-weight: 600;
            text-align: right;
        }}
        
        .solve-rate {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-weight: 600;
            text-align: center;
            padding: 6px 12px;
            border-radius: 4px;
        }}
        
        .models-list {{
            font-size: 13px;
            max-width: 300px;
            word-wrap: break-word;
        }}
        
        .attempts {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            text-align: center;
            color: #888;
        }}
        
        .stats {{
            margin-top: 20px;
            text-align: center;
            color: #888;
            font-size: 14px;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            body {{ padding: 10px; }}
            .table-container {{ font-size: 12px; }}
            th, td {{ padding: 8px 6px; }}
            .models-list {{ max-width: 150px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Task Performance Overview</h1>
        <div class="subtitle">
            {len(task_summary)} tasks analyzed{f' from {dataset_filter}' if dataset_filter else ' across all datasets'}
            • Color-coded by solve rate for easy visual scanning
        </div>
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Search tasks, datasets, or models..." 
                   onkeyup="filterTable()">
        </div>
    </div>
    
    <div class="table-container">
        <table id="taskTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)" data-type="text">Task ID<span class="sort-arrow"></span></th>
                    <th onclick="sortTable(1)" data-type="text">Dataset<span class="sort-arrow"></span></th>
                    <th onclick="sortTable(2)" data-type="number">Human Time (min)<span class="sort-arrow"></span></th>
                    <th onclick="sortTable(3)" data-type="number">Solve Rate<span class="sort-arrow"></span></th>
                    <th onclick="sortTable(4)" data-type="text">Models Solved<span class="sort-arrow"></span></th>
                    <th onclick="sortTable(5)" data-type="number">Attempts<span class="sort-arrow"></span></th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add table rows
    for task in task_summary:
        # Color-code solve rate
        solve_rate = task['solve_rate']
        if solve_rate == 0:
            color_style = "background-color: #cc3333; color: white;"
        elif solve_rate == 1:
            color_style = "background-color: #00cc66; color: black;"
        else:
            # Gradient from red to yellow to green
            if solve_rate < 0.5:
                # Red to yellow
                red = 204
                green = int(170 + (solve_rate * 2) * 85)  # 170 to 255
                blue = 51
            else:
                # Yellow to green
                red = int(255 - ((solve_rate - 0.5) * 2) * 255)  # 255 to 0
                green = 204
                blue = int(51 + ((solve_rate - 0.5) * 2) * 51)   # 51 to 102
            
            color_style = f"background-color: rgb({red}, {green}, {blue}); color: {'white' if solve_rate < 0.7 else 'black'};"
        
        html_content += f"""
                <tr>
                    <td class="task-id">{task['task_id']}</td>
                    <td class="dataset">{task['dataset']}</td>
                    <td class="human-time">{task['human_minutes']:.1f}</td>
                    <td class="solve-rate" style="{color_style}">{task['solve_percentage']}</td>
                    <td class="models-list">{task['models_list']}</td>
                    <td class="attempts">{task['successful_attempts']}/{task['total_attempts']}</td>
                </tr>"""
    
    html_content += f"""
            </tbody>
        </table>
    </div>
    
    <div class="stats">
        Generated {len(task_summary)} task summaries • 
        Click column headers to sort • 
        Use search box to filter
    </div>
    
    <script>
        let sortDirection = {{}};
        
        function sortTable(columnIndex) {{
            const table = document.getElementById('taskTable');
            const tbody = table.getElementsByTagName('tbody')[0];
            const rows = Array.from(tbody.rows);
            const th = table.getElementsByTagName('th')[columnIndex];
            const dataType = th.getAttribute('data-type');
            
            // Toggle sort direction
            const currentSort = sortDirection[columnIndex] || 'asc';
            const newSort = currentSort === 'asc' ? 'desc' : 'asc';
            sortDirection[columnIndex] = newSort;
            
            // Clear all sort indicators
            document.querySelectorAll('th').forEach(header => {{
                header.classList.remove('sort-asc', 'sort-desc');
            }});
            
            // Add sort indicator to current column
            th.classList.add(`sort-${{newSort}}`);
            
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                if (dataType === 'number') {{
                    aVal = parseFloat(aVal) || 0;
                    bVal = parseFloat(bVal) || 0;
                    return newSort === 'asc' ? aVal - bVal : bVal - aVal;
                }} else {{
                    return newSort === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }}
            }});
            
            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        function filterTable() {{
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('taskTable');
            const rows = table.getElementsByTagName('tbody')[0].rows;
            
            for (let i = 0; i < rows.length; i++) {{
                const row = rows[i];
                let showRow = false;
                
                // Search across all columns
                for (let j = 0; j < row.cells.length; j++) {{
                    const cellText = row.cells[j].textContent.toLowerCase();
                    if (cellText.includes(filter)) {{
                        showRow = true;
                        break;
                    }}
                }}
                
                row.style.display = showRow ? '' : 'none';
            }}
        }}
        
        // Sort by human time by default
        sortTable(2);
    </script>
</body>
</html>
"""
    
    # Save the HTML file
    output_file = task_analysis_dir / "task_overview.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved task overview table to {output_file}")
    logger.info(f"Generated table with {len(task_summary)} tasks")
    
    return output_file


def create_task_analysis_plots(
    output_dir: Optional[Path] = None,
    dataset_filter: Optional[str] = None
) -> List[Path]:
    """
    Create all task analysis outputs.
    
    Args:
        output_dir: Directory to save outputs
        dataset_filter: Optional dataset name to filter results
        
    Returns:
        List of paths to generated files
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "plots"
    
    generated_files = []
    
    try:
        # Generate task overview table
        table_file = create_task_overview_table(output_dir, dataset_filter)
        generated_files.append(table_file)
        
    except Exception as e:
        logger.error(f"Error generating task analysis: {e}", exc_info=True)
    
    return generated_files