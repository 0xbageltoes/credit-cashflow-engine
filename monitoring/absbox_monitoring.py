"""
AbsBox Monitoring Dashboard

This script creates a monitoring dashboard for the AbsBox service using Prometheus metrics.
It tracks performance metrics, cache hits/misses, and error rates.
"""

import sys
import os
from pathlib import Path
import time
import threading
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

# Import after setting path
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("absbox_monitoring")

# Create Flask app
app = Flask(__name__)

# Define Prometheus metrics
ABSBOX_REQUEST_COUNT = Counter('absbox_request_count', 'Number of requests to AbsBox service', ['method', 'endpoint'])
ABSBOX_REQUEST_LATENCY = Histogram('absbox_request_latency', 'AbsBox request latency in seconds', ['method', 'endpoint'])
ABSBOX_ERROR_COUNT = Counter('absbox_error_count', 'Number of errors in AbsBox service', ['method', 'error_type'])
ABSBOX_CACHE_HITS = Counter('absbox_cache_hits', 'Number of cache hits')
ABSBOX_CACHE_MISSES = Counter('absbox_cache_misses', 'Number of cache misses')
ABSBOX_ACTIVE_CALCULATIONS = Gauge('absbox_active_calculations', 'Number of active calculations')
ABSBOX_POOL_SIZE = Gauge('absbox_pool_size', 'Size of calculation thread pool')
ABSBOX_CALCULATION_QUEUE_SIZE = Gauge('absbox_calculation_queue_size', 'Size of calculation queue')

# Sample data for testing
class SampleData:
    """Generate sample data for the dashboard when real metrics are not available"""
    
    def __init__(self):
        self.start_time = datetime.now()
        
        # Initialize sample counters
        self.request_count = {
            'analyze_deal': 0,
            'run_scenario': 0,
            'health_check': 0
        }
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.active_calculations = 0
        self.pool_size = int(os.environ.get('HASTRUCTURE_MAX_POOL_SIZE', 4))
        self.queue_size = 0
        
        # Start background thread to simulate activity
        self.running = True
        self.thread = threading.Thread(target=self._simulate_activity)
        self.thread.daemon = True
        self.thread.start()
    
    def _simulate_activity(self):
        """Simulate AbsBox activity in a background thread"""
        while self.running:
            # Simulate new requests
            if np.random.random() < 0.3:  # 30% chance of a new request
                request_type = np.random.choice(['analyze_deal', 'run_scenario', 'health_check'], 
                                             p=[0.6, 0.3, 0.1])  # Different probabilities for each type
                self.request_count[request_type] += 1
                
                # Simulate errors (10% chance)
                if np.random.random() < 0.1:
                    self.error_count += 1
                
                # Simulate cache activity
                if np.random.random() < 0.7:  # 70% cache hit rate
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                
                # Simulate active calculations
                self.active_calculations = min(self.pool_size, self.active_calculations + 1)
                self.queue_size = max(0, self.queue_size - 1) if self.queue_size > 0 else 0
            else:
                # Simulate completed calculations
                self.active_calculations = max(0, self.active_calculations - 1)
                
                # Simulate new items in the queue
                if np.random.random() < 0.2:
                    self.queue_size += np.random.randint(1, 3)
            
            # Sleep for a random time
            time.sleep(np.random.uniform(0.5, 2.0))
    
    def get_metrics(self):
        """Get the current metrics"""
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime': str(uptime).split('.')[0],  # Remove microseconds
            'request_count': self.request_count,
            'total_requests': sum(self.request_count.values()),
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, sum(self.request_count.values())) * 100,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100,
            'active_calculations': self.active_calculations,
            'pool_size': self.pool_size,
            'queue_size': self.queue_size
        }
    
    def stop(self):
        """Stop the simulation thread"""
        self.running = False
        self.thread.join()

# Create sample data instance
sample_data = SampleData()

@app.route('/')
def index():
    """Render the dashboard"""
    metrics = sample_data.get_metrics()
    return render_template('absbox_dashboard.html', metrics=metrics)

@app.route('/metrics')
def metrics():
    """Endpoint for Prometheus metrics"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for dashboard metrics"""
    metrics = sample_data.get_metrics()
    return jsonify(metrics)

@app.route('/api/historical')
def historical_metrics():
    """API endpoint for historical metrics"""
    # Generate some historical data
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i*5) for i in range(24)]  # Last 2 hours in 5-minute intervals
    timestamps.reverse()  # Oldest first
    
    # Generate random data with some patterns
    requests = [int(50 + 30 * np.sin(i/4) + np.random.randint(-10, 10)) for i in range(24)]
    errors = [int(max(0, r * 0.1 * np.random.random())) for r in requests]
    cache_hit_rate = [70 + 10 * np.sin(i/6) + np.random.randint(-5, 5) for i in range(24)]
    
    # Format for the frontend
    historical_data = {
        'timestamps': [ts.strftime('%H:%M') for ts in timestamps],
        'requests': requests,
        'errors': errors,
        'cache_hit_rate': cache_hit_rate
    }
    
    return jsonify(historical_data)

def create_html_template():
    """Create the HTML template for the dashboard"""
    template_path = Path(__file__).parent / 'templates'
    template_path.mkdir(exist_ok=True)
    
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AbsBox Monitoring Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { 
                padding-top: 20px; 
                background-color: #f5f5f5;
            }
            .card {
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .card-header {
                font-weight: bold;
                background-color: #f8f9fa;
            }
            .dashboard-title {
                margin-bottom: 30px;
                color: #333;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
            }
            .metric-label {
                font-size: 14px;
                color: #6c757d;
            }
            .chart-container {
                position: relative;
                height: 250px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center dashboard-title">AbsBox Monitoring Dashboard</h1>
            
            <div class="row">
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-header">Uptime</div>
                        <div class="card-body text-center">
                            <div class="metric-value">{{ metrics.uptime }}</div>
                            <div class="metric-label">HH:MM:SS</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-header">Total Requests</div>
                        <div class="card-body text-center">
                            <div class="metric-value">{{ metrics.total_requests }}</div>
                            <div class="metric-label">Requests</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-header">Error Rate</div>
                        <div class="card-body text-center">
                            <div class="metric-value">{{ "%.2f"|format(metrics.error_rate) }}%</div>
                            <div class="metric-label">{{ metrics.error_count }} Errors</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-header">Cache Hit Rate</div>
                        <div class="card-body text-center">
                            <div class="metric-value">{{ "%.2f"|format(metrics.cache_hit_rate) }}%</div>
                            <div class="metric-label">{{ metrics.cache_hits }} Hits / {{ metrics.cache_misses }} Misses</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Active Calculations</div>
                        <div class="card-body">
                            <div class="progress" style="height: 30px;">
                                <div class="progress-bar" role="progressbar" 
                                    style="width: {{ (metrics.active_calculations / metrics.pool_size * 100) | int }}%;" 
                                    aria-valuenow="{{ metrics.active_calculations }}" 
                                    aria-valuemin="0" 
                                    aria-valuemax="{{ metrics.pool_size }}">
                                    {{ metrics.active_calculations }} / {{ metrics.pool_size }}
                                </div>
                            </div>
                            <div class="metric-label mt-2">Thread Pool Utilization</div>
                            
                            <div class="mt-3">
                                <span class="metric-label">Queue Size: </span>
                                <span class="metric-value" style="font-size: 18px;">{{ metrics.queue_size }}</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">Request Distribution</div>
                        <div class="card-body">
                            <canvas id="requestChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">Historical Metrics</div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="historyChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="mt-4 mb-4 text-center text-muted">
                <small>AbsBox Monitoring Dashboard | Last Updated: <span id="lastUpdated"></span></small>
                <br>
                <small>Data refreshes automatically every 5 seconds</small>
            </footer>
        </div>
        
        <script>
            // Initialize request distribution chart
            const requestCtx = document.getElementById('requestChart').getContext('2d');
            const requestChart = new Chart(requestCtx, {
                type: 'pie',
                data: {
                    labels: ['Analyze Deal', 'Run Scenario', 'Health Check'],
                    datasets: [{
                        data: [
                            {{ metrics.request_count.analyze_deal }}, 
                            {{ metrics.request_count.run_scenario }}, 
                            {{ metrics.request_count.health_check }}
                        ],
                        backgroundColor: [
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 206, 86, 0.7)',
                            'rgba(75, 192, 192, 0.7)'
                        ],
                        borderColor: [
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                        }
                    }
                }
            });
            
            // Initialize historical metrics chart
            let historyChart;
            
            // Function to update the historical chart
            function updateHistoricalChart() {
                fetch('/api/historical')
                    .then(response => response.json())
                    .then(data => {
                        if (historyChart) {
                            historyChart.destroy();
                        }
                        
                        const historyCtx = document.getElementById('historyChart').getContext('2d');
                        historyChart = new Chart(historyCtx, {
                            type: 'line',
                            data: {
                                labels: data.timestamps,
                                datasets: [{
                                    label: 'Requests',
                                    data: data.requests,
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                    yAxisID: 'y',
                                    tension: 0.3
                                }, {
                                    label: 'Errors',
                                    data: data.errors,
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                    yAxisID: 'y',
                                    tension: 0.3
                                }, {
                                    label: 'Cache Hit Rate (%)',
                                    data: data.cache_hit_rate,
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                                    yAxisID: 'y1',
                                    tension: 0.3
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        type: 'linear',
                                        display: true,
                                        position: 'left',
                                        title: {
                                            display: true,
                                            text: 'Count'
                                        }
                                    },
                                    y1: {
                                        type: 'linear',
                                        display: true,
                                        position: 'right',
                                        title: {
                                            display: true,
                                            text: 'Percentage'
                                        },
                                        min: 0,
                                        max: 100,
                                        grid: {
                                            drawOnChartArea: false
                                        }
                                    }
                                }
                            }
                        });
                    });
            }
            
            // Update dashboard data
            function updateDashboard() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        // Update metrics
                        document.querySelector('.col-md-3:nth-child(1) .metric-value').textContent = data.uptime;
                        document.querySelector('.col-md-3:nth-child(2) .metric-value').textContent = data.total_requests;
                        document.querySelector('.col-md-3:nth-child(3) .metric-value').textContent = data.error_rate.toFixed(2) + '%';
                        document.querySelector('.col-md-3:nth-child(3) .metric-label').textContent = data.error_count + ' Errors';
                        document.querySelector('.col-md-3:nth-child(4) .metric-value').textContent = data.cache_hit_rate.toFixed(2) + '%';
                        document.querySelector('.col-md-3:nth-child(4) .metric-label').textContent = 
                            data.cache_hits + ' Hits / ' + data.cache_misses + ' Misses';
                        
                        // Update progress bar
                        const progressBar = document.querySelector('.progress-bar');
                        const progressPercentage = (data.active_calculations / data.pool_size * 100).toFixed(0) + '%';
                        progressBar.style.width = progressPercentage;
                        progressBar.setAttribute('aria-valuenow', data.active_calculations);
                        progressBar.textContent = data.active_calculations + ' / ' + data.pool_size;
                        
                        // Update queue size
                        document.querySelector('.mt-3 .metric-value').textContent = data.queue_size;
                        
                        // Update request chart
                        requestChart.data.datasets[0].data = [
                            data.request_count.analyze_deal,
                            data.request_count.run_scenario,
                            data.request_count.health_check
                        ];
                        requestChart.update();
                        
                        // Update last updated time
                        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
                    });
            }
            
            // Initial chart load
            updateHistoricalChart();
            
            // Set up auto-refresh
            setInterval(updateDashboard, 5000);
            setInterval(updateHistoricalChart, 60000);
            
            // Initial timestamp
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
        </script>
    </body>
    </html>
    """
    
    with open(template_path / 'absbox_dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    logger.info(f"Created dashboard template at {template_path / 'absbox_dashboard.html'}")

def main():
    """Main function to start the monitoring dashboard"""
    logger.info("Starting AbsBox Monitoring Dashboard")
    
    # Create HTML template
    create_html_template()
    
    # Get port from environment or use default
    port = int(os.environ.get('ABSBOX_DASHBOARD_PORT', 5000))
    
    try:
        logger.info(f"Dashboard running at http://localhost:{port}")
        app.run(host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard")
        sample_data.stop()
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        sample_data.stop()

if __name__ == "__main__":
    main()
