#!/usr/bin/env python3
import subprocess
import threading
import time
import curses
import signal
import sys
import psutil
from datetime import datetime

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.process_counter = 0
        self.selected_process = 0
        self.view_mode = "status"
        self.selected_log_process = 0
        
        # Base script template
        self.script_template = '''
export HF_ENDPOINT=https://hf-mirror.com/
cd /workspace/VADAR
. venv2/bin/activate
cd demo-notebook
python processing_client.py --server_url http://0.0.0.0:8000 --mistral_url http://0.0.0.0:8001 --image_dir /workspace/PhysicalAI_Dataset/train_sample
'''

    def spawn_process(self, custom_args=""):
        """Spawn a new process with optional custom arguments"""
        self.process_counter += 1
        process_id = f"proc_{self.process_counter}"
        
        # Modify script if custom args provided
        script = self.script_template.strip()
        if custom_args:
            lines = script.split('\n')
            lines[-1] = f"python processing_client.py {custom_args}"
            script = '\n'.join(lines)
        
        # Create lightweight process info
        process_info = {
            'id': process_id,
            'args': custom_args if custom_args else "default",
            'process': None,
            'status': 'Starting',
            'start_time': datetime.now(),
            'pid': None,
            'cpu_percent': 0.0,
            'memory_mb': 0.0
        }
        
        try:
            # Start the process with minimal overhead
            process = subprocess.Popen(
                ['bash', '-c', script],
                stdout=subprocess.DEVNULL,  # Discard output for performance
                stderr=subprocess.DEVNULL,  # Discard errors for performance
                preexec_fn=None
            )
            
            process_info['process'] = process
            process_info['pid'] = process.pid
            process_info['status'] = 'Running'
            
            self.processes[process_id] = process_info
            
        except Exception as e:
            process_info['status'] = f'Failed: {str(e)}'
            self.processes[process_id] = process_info

    def kill_process(self, process_id):
        """Kill a specific process and its children"""
        if process_id in self.processes:
            process_info = self.processes[process_id]
            if process_info['process'] and process_info['process'].poll() is None:
                try:
                    # Kill process tree to ensure clean shutdown
                    parent = psutil.Process(process_info['pid'])
                    children = parent.children(recursive=True)
                    for child in children:
                        child.terminate()
                    parent.terminate()
                    
                    # Wait a bit, then force kill if needed
                    time.sleep(1)
                    for child in children:
                        if child.is_running():
                            child.kill()
                    if parent.is_running():
                        parent.kill()
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process already dead or no permission
                    pass
                    
            process_info['status'] = 'Killed'

    def update_process_status(self):
        """Lightweight status update"""
        for process_id, process_info in list(self.processes.items()):
            if process_info['process']:
                # Check if process is still running
                if process_info['process'].poll() is not None:
                    if process_info['status'] == 'Running':
                        return_code = process_info['process'].returncode
                        if return_code == 0:
                            process_info['status'] = 'Completed'
                        else:
                            process_info['status'] = f'Failed (code: {return_code})'
                        process_info['cpu_percent'] = 0.0
                        process_info['memory_mb'] = 0.0
                elif process_info['status'] == 'Running' and process_info['pid']:
                    # Get lightweight system stats
                    try:
                        proc = psutil.Process(process_info['pid'])
                        process_info['cpu_percent'] = proc.cpu_percent()
                        process_info['memory_mb'] = proc.memory_info().rss / 1024 / 1024
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        process_info['status'] = 'Dead'
                        process_info['cpu_percent'] = 0.0
                        process_info['memory_mb'] = 0.0

    def get_live_tail(self, process_id, lines=20):
        """Get live tail of process output (only when explicitly requested)"""
        if process_id not in self.processes:
            return ["Process not found"]
        
        process_info = self.processes[process_id]
        if not process_info['process'] or process_info['process'].poll() is not None:
            return ["Process not running"]
        
        try:
            # Use tail command to get recent output without storing it
            pid = process_info['pid']
            # This is a simplified approach - in production you might want to 
            # temporarily capture output only when viewing logs
            return [f"Live output view not implemented for performance reasons.",
                   f"Process PID: {pid}",
                   f"Status: {process_info['status']}",
                   f"Use 'tail -f /proc/{pid}/fd/1' in another terminal to view output"]
        except Exception as e:
            return [f"Error getting output: {str(e)}"]

class Dashboard:
    def __init__(self):
        self.pm = ProcessManager()
        self.running = True
        self.last_update = 0
        self.update_interval = 1.0  # Update every 1 second instead of 100ms
        
    def draw_status_view(self, stdscr):
        """Draw the lightweight status view"""
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        
        # Title
        title = "VADAR Process Manager Dashboard (Lightweight)"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        
        # System info
        try:
            cpu_total = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            system_info = f"System CPU: {cpu_total:.1f}% | RAM: {mem.percent:.1f}% ({mem.used//1024//1024}MB/{mem.total//1024//1024}MB)"
            stdscr.addstr(1, 2, system_info)
        except:
            pass
        
        # Instructions
        instructions = [
            "Commands: [s]pawn [k]ill [l]ogs [c]lear dead [q]uit [↑↓] navigate",
            "Note: Output not tracked for performance - use external terminal for logs"
        ]
        
        for i, instruction in enumerate(instructions):
            stdscr.addstr(3 + i, 2, instruction)
        
        # Headers
        header_y = 6
        headers = ["ID", "Status", "PID", "CPU%", "RAM(MB)", "Runtime", "Args"]
        col_widths = [10, 12, 8, 8, 10, 12, 20]
        
        x_pos = 2
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            stdscr.addstr(header_y, x_pos, header.ljust(width), curses.A_UNDERLINE)
            x_pos += width + 1
        
        # Process list
        processes = list(self.pm.processes.items())
        start_y = header_y + 2
        
        for i, (proc_id, proc_info) in enumerate(processes):
            y_pos = start_y + i
            if y_pos >= height - 2:
                break
                
            # Highlight selected process
            attr = curses.A_REVERSE if i == self.pm.selected_process else curses.A_NORMAL
            
            # Calculate runtime
            runtime = datetime.now() - proc_info['start_time']
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            
            # Format data
            data = [
                proc_id[:9],
                proc_info['status'][:11],
                str(proc_info['pid'] if proc_info['pid'] else 'N/A')[:7],
                f"{proc_info['cpu_percent']:.1f}"[:7],
                f"{proc_info['memory_mb']:.0f}"[:9],
                runtime_str[:11],
                proc_info['args'][:19]
            ]
            
            x_pos = 2
            for j, (item, width) in enumerate(zip(data, col_widths)):
                stdscr.addstr(y_pos, x_pos, item.ljust(width), attr)
                x_pos += width + 1
        
        # Status bar
        running_count = sum(1 for p in self.pm.processes.values() if p['status'] == 'Running')
        status_text = f"Total: {len(self.pm.processes)} | Running: {running_count} | Press 'h' for help"
        stdscr.addstr(height - 1, 2, status_text)
        
    def draw_logs_view(self, stdscr):
        """Draw minimal logs view"""
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        
        processes = list(self.pm.processes.items())
        if not processes:
            stdscr.addstr(height // 2, (width - 20) // 2, "No processes running")
            stdscr.addstr(height - 1, 2, "Press 'b' to go back")
            return
        
        if self.pm.selected_log_process >= len(processes):
            self.pm.selected_log_process = len(processes) - 1
        
        proc_id, proc_info = processes[self.pm.selected_log_process]
        
        # Title
        title = f"Process Info: {proc_id} ({proc_info['status']})"
        stdscr.addstr(0, 2, title, curses.A_BOLD)
        
        # Instructions
        instructions = "Commands: [b]ack [↑↓] switch processes [t]erminate [q]uit"
        stdscr.addstr(1, 2, instructions)
        
        # Process details
        details = [
            f"PID: {proc_info['pid']}",
            f"Status: {proc_info['status']}",
            f"Started: {proc_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"Arguments: {proc_info['args']}",
            f"CPU Usage: {proc_info['cpu_percent']:.1f}%",
            f"Memory Usage: {proc_info['memory_mb']:.0f} MB",
            "",
            "For live output, use in another terminal:",
            f"  tail -f /proc/{proc_info['pid']}/fd/1    # stdout",
            f"  tail -f /proc/{proc_info['pid']}/fd/2    # stderr",
            "",
            "Or monitor with:",
            f"  watch -n 1 'ps aux | grep {proc_info['pid']}'",
            "",
            "Output tracking disabled for performance optimization."
        ]
        
        for i, detail in enumerate(details):
            y_pos = 3 + i
            if y_pos < height - 2:
                stdscr.addstr(y_pos, 2, detail)
        
        # Status bar
        status_text = f"Process {self.pm.selected_log_process + 1}/{len(processes)}"
        stdscr.addstr(height - 1, 2, status_text)
    
    def get_custom_args(self, stdscr):
        """Get custom arguments for spawning process"""
        height, width = stdscr.getmaxyx()
        
        input_height = 5
        input_width = min(80, width - 4)
        input_y = (height - input_height) // 2
        input_x = (width - input_width) // 2
        
        input_win = curses.newwin(input_height, input_width, input_y, input_x)
        input_win.box()
        input_win.addstr(1, 2, "Enter custom args (or Enter for default):")
        input_win.addstr(2, 2, "Args: ")
        input_win.refresh()
        
        curses.curs_set(1)
        curses.echo()
        
        args = input_win.getstr(2, 8, input_width - 10).decode('utf-8')
        
        curses.curs_set(0)
        curses.noecho()
        
        return args.strip()
    
    def show_help(self, stdscr):
        """Show help screen"""
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        
        help_text = [
            "VADAR Process Manager - Lightweight Version",
            "",
            "Status View Commands:",
            "  s - Spawn new process (with optional custom args)",
            "  k - Kill selected process (including child processes)",
            "  l - View process info and monitoring tips",
            "  c - Clear dead/completed processes from list",
            "  ↑/↓ - Navigate process list",
            "  q - Quit application",
            "",
            "Process Info View Commands:",
            "  b - Back to status view",
            "  ↑/↓ - Switch between processes",
            "  t - Terminate current process",
            "",
            "Performance Notes:",
            "  - Output is not captured to avoid CPU overhead",
            "  - Use external terminals for live output monitoring",
            "  - Process stats updated every 1 second",
            "  - Lightweight psutil monitoring only",
            "",
            "Monitoring Tips:",
            "  tail -f /proc/PID/fd/1    # View stdout",
            "  htop -p PID1,PID2,PID3   # Monitor multiple processes",
            "",
            "Press any key to continue..."
        ]
        
        for i, line in enumerate(help_text):
            if i < height - 1:
                stdscr.addstr(i, 2, line)
        
        stdscr.getch()
    
    def run(self, stdscr):
        """Main application loop with reduced refresh rate"""
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.timeout(200)  # 200ms timeout for more responsive input
        
        while self.running:
            try:
                current_time = time.time()
                
                # Only update process status every second to reduce CPU usage
                if current_time - self.last_update > self.update_interval:
                    self.pm.update_process_status()
                    self.last_update = current_time
                
                # Draw current view
                if self.pm.view_mode == "status":
                    self.draw_status_view(stdscr)
                elif self.pm.view_mode == "logs":
                    self.draw_logs_view(stdscr)
                
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                
                if key == ord('q'):
                    self.running = False
                elif key == ord('h'):
                    self.show_help(stdscr)
                elif self.pm.view_mode == "status":
                    self.handle_status_keys(key, stdscr)
                elif self.pm.view_mode == "logs":
                    self.handle_logs_keys(key)
                    
            except KeyboardInterrupt:
                self.running = False
        
        # Clean up processes
        for proc_id in list(self.pm.processes.keys()):
            self.pm.kill_process(proc_id)
    
    def handle_status_keys(self, key, stdscr):
        """Handle keys in status view"""
        processes = list(self.pm.processes.items())
        
        if key == curses.KEY_UP and self.pm.selected_process > 0:
            self.pm.selected_process -= 1
        elif key == curses.KEY_DOWN and self.pm.selected_process < len(processes) - 1:
            self.pm.selected_process += 1
        elif key == ord('s'):
            custom_args = self.get_custom_args(stdscr)
            self.pm.spawn_process(custom_args)
        elif key == ord('k') and processes:
            if self.pm.selected_process < len(processes):
                proc_id = processes[self.pm.selected_process][0]
                self.pm.kill_process(proc_id)
        elif key == ord('l') and processes:
            self.pm.view_mode = "logs"
            self.pm.selected_log_process = self.pm.selected_process
        elif key == ord('c'):
            # Clear dead processes
            dead_procs = [pid for pid, info in self.pm.processes.items() 
                         if info['status'] not in ['Running', 'Starting']]
            for pid in dead_procs:
                del self.pm.processes[pid]
            if self.pm.selected_process >= len(self.pm.processes):
                self.pm.selected_process = max(0, len(self.pm.processes) - 1)
    
    def handle_logs_keys(self, key):
        """Handle keys in logs view"""
        processes = list(self.pm.processes.items())
        
        if key == ord('b'):
            self.pm.view_mode = "status"
        elif key == ord('t') and processes:
            proc_id = processes[self.pm.selected_log_process][0]
            self.pm.kill_process(proc_id)
        elif key == curses.KEY_UP and self.pm.selected_log_process > 0:
            self.pm.selected_log_process -= 1
        elif key == curses.KEY_DOWN and self.pm.selected_log_process < len(processes) - 1:
            self.pm.selected_log_process += 1

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    dashboard = Dashboard()
    try:
        curses.wrapper(dashboard.run)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up any remaining processes
        for proc_id in list(dashboard.pm.processes.keys()):
            dashboard.pm.kill_process(proc_id)
        print("Dashboard closed. All processes terminated.")

if __name__ == "__main__":
    main()