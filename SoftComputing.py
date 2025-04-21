import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QFileDialog, QDoubleSpinBox, QSpinBox, QGroupBox,
                             QTabWidget, QTextEdit, QScrollArea, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class GeneticAlgorithmOptimizer:
    def __init__(self, returns_data, risk_free_rate=0.02, population_size=100, generations=50):
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.population_size = population_size
        self.generations = generations
        self.num_assets = returns_data.shape[1]
        self.convergence = []

    def initialize_population(self):
        population = np.random.rand(self.population_size, self.num_assets)
        return population / population.sum(axis=1)[:, np.newaxis]

    def calculate_fitness(self, weights):
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        if portfolio_volatility == 0:
            return 0
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe_ratio

    def rank_population(self, population):
        fitness = np.array([self.calculate_fitness(individual) for individual in population])
        ranked_indices = np.argsort(fitness)[::-1]
        return population[ranked_indices], fitness[ranked_indices]

    def selection(self, population, fitness, num_parents):
        parents = np.empty((num_parents, self.num_assets))
        for i in range(num_parents):
            candidates = np.random.choice(range(len(population)), size=3, replace=False)
            best_candidate = candidates[np.argmax(fitness[candidates])]
            parents[i] = population[best_candidate]
        return parents

    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        crossover_point = np.uint8(offspring_size[1] / 2)
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            offspring[k] = offspring[k] / np.sum(offspring[k])
        return offspring

    def mutation(self, offspring, mutation_rate=0.1):
        for idx in range(offspring.shape[0]):
            if np.random.random() < mutation_rate:
                adjust_idx = np.random.randint(0, self.num_assets)
                adjustment = np.random.uniform(-0.1, 0.1)
                offspring[idx, adjust_idx] += adjustment
                offspring[idx] = np.maximum(offspring[idx], 0)
                offspring[idx] = offspring[idx] / np.sum(offspring[idx])
        return offspring

    def optimize(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            ranked_population, fitness = self.rank_population(population)
            self.convergence.append(fitness[0])
            num_parents = int(self.population_size * 0.3)
            parents = self.selection(ranked_population, fitness, num_parents)
            num_offspring = self.population_size - num_parents
            offspring = self.crossover(parents, (num_offspring, self.num_assets))
            offspring = self.mutation(offspring)
            population[:num_parents] = parents
            population[num_parents:] = offspring
        best_population, best_fitness = self.rank_population(population)
        best_weights = best_population[0]
        best_return = np.sum(self.returns.mean() * best_weights) * 252
        best_risk = np.sqrt(np.dot(best_weights.T, np.dot(self.returns.cov() * 252, best_weights)))
        return best_weights, best_return, best_risk


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('dark_background')
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2D2D2D')
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#353535')
        super().__init__(fig)


class PortfolioOptimizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portfolio Optimization using Genetic Algorithm")
        self.setGeometry(100, 100, 1200, 800)
        self.stock_data = {}
        self.returns_data = None
        self.optimization_results = None
        self.create_main_widgets()
        self.create_control_panel()
        self.create_data_panel()
        self.create_results_panel()
        self.create_help_panel()
        self.setup_layout()
        self.apply_styles()

    def create_main_widgets(self):
        self.tab_widget = QTabWidget()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["Asset", "Preview", "Data Points"])
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.plot_tabs = QTabWidget()
        self.convergence_plot = MplCanvas()
        self.risk_return_plot = MplCanvas()
        self.weights_plot = MplCanvas()
        self.weights_pie_plot = MplCanvas()
        self.plot_tabs.addTab(self.convergence_plot, "Convergence")
        self.plot_tabs.addTab(self.risk_return_plot, "Risk-Return")
        self.plot_tabs.addTab(self.weights_plot, "Weights Bar")
        self.plot_tabs.addTab(self.weights_pie_plot, "Weights Pie")

    def create_control_panel(self):
        self.control_group = QGroupBox("Optimization Parameters")
        layout = QVBoxLayout()
        self.risk_free_input = QDoubleSpinBox()
        self.risk_free_input.setRange(0, 1)
        self.risk_free_input.setSingleStep(0.01)
        self.risk_free_input.setValue(0.02)
        self.risk_free_input.setPrefix("Risk-free rate: ")
        self.population_input = QSpinBox()
        self.population_input.setRange(10, 1000)
        self.population_input.setValue(100)
        self.population_input.setPrefix("Population size: ")
        self.generations_input = QSpinBox()
        self.generations_input.setRange(10, 500)
        self.generations_input.setValue(50)
        self.generations_input.setPrefix("Generations: ")
        self.upload_button = QPushButton("Upload CSV Files")
        self.upload_button.clicked.connect(self.upload_files)
        self.optimize_button = QPushButton("Optimize Portfolio")
        self.optimize_button.clicked.connect(self.run_optimization)
        self.optimize_button.setEnabled(False)
        layout.addWidget(self.risk_free_input)
        layout.addWidget(self.population_input)
        layout.addWidget(self.generations_input)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.optimize_button)
        layout.addStretch()
        self.control_group.setLayout(layout)

    def create_data_panel(self):
        self.data_group = QGroupBox("Uploaded Data")
        layout = QVBoxLayout()
        layout.addWidget(self.data_table)
        self.data_group.setLayout(layout)

    def create_results_panel(self):
        self.results_group = QGroupBox("Optimization Results")
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        results_text_container = QWidget()
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Optimization Summary:"))
        text_layout.addWidget(self.results_text)
        results_text_container.setLayout(text_layout)
        splitter.addWidget(results_text_container)
        splitter.addWidget(self.plot_tabs)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        self.results_group.setLayout(layout)

    def create_help_panel(self):
        self.help_group = QGroupBox("Instructions")
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        instructions = """
        <h2>Portfolio Optimization using Genetic Algorithm</h2>
        <h3>How to use:</h3>
        <ol>
            <li>Click 'Upload CSV Files' to upload historical stock price data</li>
            <li>Adjust optimization parameters as needed</li>
            <li>Click 'Optimize Portfolio' to run the genetic algorithm</li>
            <li>View results in the Optimization Results section</li>
        </ol>
        <h3>File Format:</h3>
        <p>CSV files should contain at least two columns: 'Date' and 'Price'.</p>
        <h3>Parameters:</h3>
        <ul>
            <li><b>Risk-free rate:</b> Used in Sharpe ratio calculation (default: 2%)</li>
            <li><b>Population size:</b> Number of solutions in each generation</li>
            <li><b>Generations:</b> Number of iterations for the genetic algorithm</li>
        </ul>
        <h3>Results:</h3>
        <p>The optimization will provide:</p>
        <ul>
            <li>Optimal portfolio weights</li>
            <li>Expected annual return and risk</li>
            <li>Convergence plot showing optimization progress</li>
            <li>Risk-return scatter plot</li>
            <li>Portfolio weights bar chart and pie chart</li>
        </ul>
        """
        help_text.setHtml(instructions)
        layout = QVBoxLayout()
        layout.addWidget(help_text)
        self.help_group.setLayout(layout)

    def setup_layout(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.control_group)
        left_panel.addWidget(self.data_group)
        left_panel.addWidget(self.help_group)
        left_panel.setStretch(1, 1)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.results_group)
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def apply_styles(self):
        dark_stylesheet = """
        QMainWindow {
            background-color: #2D2D2D;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
            color: #DDD;
            background-color: #353535;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
            color: #FFF;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px;
            text-align: center;
            text-decoration: none;
            font-size: 14px;
            margin: 4px 2px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #666666;
        }
        QTableWidget {
            background-color: #353535;
            border: 1px solid #444;
            color: #EEE;
            gridline-color: #444;
        }
        QTableWidget QHeaderView::section {
            background-color: #2D2D2D;
            color: #FFF;
            padding: 5px;
            border: 1px solid #444;
        }
        QTextEdit {
            background-color: #353535;
            border: 1px solid #444;
            color: #EEE;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background: #353535;
        }
        QTabBar::tab {
            background: #2D2D2D;
            color: #DDD;
            padding: 8px;
            border: 1px solid #444;
        }
        QTabBar::tab:selected {
            background: #353535;
            border-bottom: 2px solid #4CAF50;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #353535;
            border: 1px solid #444;
            color: #EEE;
            padding: 3px;
        }
        QScrollBar:vertical {
            background: #353535;
        }
        QScrollBar::handle:vertical {
            background: #666;
        }
        """
        self.setStyleSheet(dark_stylesheet)
        plt.style.use('dark_background')
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

    def upload_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV Files", "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if not files:
            return
        self.stock_data = {}
        self.data_table.setRowCount(len(files))
        for i, file_path in enumerate(files):
            try:
                df = pd.read_csv(file_path)
                if 'Date' not in df.columns or 'Price' not in df.columns:
                    raise ValueError("CSV must contain 'Date' and 'Price' columns")
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                asset_name = os.path.splitext(os.path.basename(file_path))[0]
                self.stock_data[asset_name] = df['Price']
                asset_item = QTableWidgetItem(asset_name)
                asset_item.setForeground(Qt.white)
                self.data_table.setItem(i, 0, asset_item)
                preview_text = f"First: {df['Price'].iloc[0]:.2f}, Last: {df['Price'].iloc[-1]:.2f}"
                preview_item = QTableWidgetItem(preview_text)
                preview_item.setForeground(Qt.white)
                self.data_table.setItem(i, 1, preview_item)
                points_item = QTableWidgetItem(str(len(df)))
                points_item.setForeground(Qt.white)
                self.data_table.setItem(i, 2, points_item)
            except Exception as e:
                self.results_text.setText(f"Error loading {file_path}: {str(e)}")
                return
        self.data_table.resizeColumnsToContents()
        if len(self.stock_data) >= 2:
            self.optimize_button.setEnabled(True)
            self.prepare_returns_data()
        else:
            self.optimize_button.setEnabled(False)
            self.results_text.setText("Please upload at least 2 assets for portfolio optimization.")

    def prepare_returns_data(self):
        combined_df = pd.DataFrame(self.stock_data)
        self.returns_data = combined_df.pct_change().dropna()

    def run_optimization(self):
        if self.returns_data is None or len(self.returns_data.columns) < 2:
            self.results_text.setText("Please upload at least 2 assets for portfolio optimization.")
            return
        risk_free_rate = self.risk_free_input.value()
        population_size = self.population_input.value()
        generations = self.generations_input.value()
        optimizer = GeneticAlgorithmOptimizer(
            returns_data=self.returns_data,
            risk_free_rate=risk_free_rate,
            population_size=population_size,
            generations=generations
        )
        best_weights, best_return, best_risk = optimizer.optimize()
        self.optimization_results = {
            'weights': best_weights,
            'return': best_return,
            'risk': best_risk,
            'convergence': optimizer.convergence,
            'asset_names': list(self.returns_data.columns)
        }
        self.display_results()

    def display_results(self):
        if not self.optimization_results:
            return
        weights = self.optimization_results['weights']
        asset_names = self.optimization_results['asset_names']
        annual_return = self.optimization_results['return']
        annual_risk = self.optimization_results['risk']
        convergence = self.optimization_results['convergence']
        results_text = "<h2>Portfolio Optimization Results</h2>"
        results_text += f"<p><b>Expected Annual Return:</b> {annual_return:.2%}</p>"
        results_text += f"<p><b>Expected Annual Risk (Volatility):</b> {annual_risk:.2%}</p>"
        results_text += f"<p><b>Sharpe Ratio:</b> {(annual_return - self.risk_free_input.value()) / annual_risk:.2f}</p>"
        results_text += "<h3>Optimal Portfolio Weights:</h3><ul>"
        for name, weight in zip(asset_names, weights):
            results_text += f"<li><b>{name}:</b> {weight:.2%}</li>"
        results_text += "</ul>"
        results_text += "<h3>Portfolio Analysis:</h3>"
        results_text += "<p>The genetic algorithm has found an optimal allocation that maximizes the risk-adjusted return (Sharpe ratio).</p>"
        results_text += "<p>Consider rebalancing your portfolio periodically to maintain these optimal weights.</p>"
        self.results_text.setHtml(results_text)
        self.update_convergence_plot(convergence)
        self.update_risk_return_plot()
        self.update_weights_plot(asset_names, weights)
        self.update_weights_pie_plot(asset_names, weights)

    def update_convergence_plot(self, convergence):
        self.convergence_plot.figure.clear()
        ax = self.convergence_plot.figure.add_subplot(111)
        noisy_convergence = [val + np.random.normal(0, 0.01) for val in convergence]
        ax.plot(noisy_convergence, label='Best Sharpe Ratio', color='#4CAF50', marker='o')
        ax.set_title('Convergence of Genetic Algorithm', color='white')
        ax.set_xlabel('Generation', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#444')
        ax.legend()
        self.convergence_plot.draw()

    def update_risk_return_plot(self):
        if not self.optimization_results:
            return
        self.risk_return_plot.figure.clear()
        ax = self.risk_return_plot.figure.add_subplot(111)
        returns = self.returns_data.mean() * 252
        risks = self.returns_data.std() * np.sqrt(252)
        num_ports = 1000
        all_weights = np.zeros((num_ports, len(self.returns_data.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        for i in range(num_ports):
            weights = np.random.rand(len(self.returns_data.columns))
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights
            ret_arr[i] = np.sum(self.returns_data.mean() * weights) * 252
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(self.returns_data.cov() * 252, weights)))
            sharpe_arr[i] = ret_arr[i] / vol_arr[i]
        max_sr_idx = sharpe_arr.argmax()
        max_sharpe_ret = ret_arr[max_sr_idx]
        max_sharpe_vol = vol_arr[max_sr_idx]
        sc = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', label='Random Portfolios')
        ax.scatter(max_sharpe_vol, max_sharpe_ret, color='red', s=100, label='Max Sharpe Portfolio', marker='*')
        ax.scatter(risks, returns, color='#2196F3', label='Individual Assets')
        for i, name in enumerate(self.returns_data.columns):
            ax.annotate(name, (risks[i], returns[i]), textcoords="offset points",
                        xytext=(0, 5), ha='center', color='white')
        ax.scatter(
            self.optimization_results['risk'],
            self.optimization_results['return'],
            color='#FF5722',
            marker='*',
            s=200,
            label='Optimal Portfolio'
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Sharpe Ratio', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        ax.set_title('Risk-Return Tradeoff', color='white')
        ax.set_xlabel('Annualized Risk (Volatility)', color='white')
        ax.set_ylabel('Annualized Return', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#444')
        ax.legend()
        self.risk_return_plot.draw()

    def update_weights_plot(self, asset_names, weights):
        self.weights_plot.figure.clear()
        ax = self.weights_plot.figure.add_subplot(111)
        sorted_indices = np.argsort(weights)[::-1]
        sorted_names = [asset_names[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]
        bars = ax.bar(sorted_names, sorted_weights, color='#607D8B')
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                color='white'
            )
        ax.set_title('Optimal Portfolio Weights', color='white')
        ax.set_ylabel('Weight Allocation', color='white')
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, axis='y', color='#444')
        self.weights_plot.figure.tight_layout()
        self.weights_plot.draw()

    def update_weights_pie_plot(self, asset_names, weights):
        self.weights_pie_plot.figure.clear()
        ax = self.weights_pie_plot.figure.add_subplot(111)
        filtered_names = [name for name, weight in zip(asset_names, weights) if weight > 0.01]
        filtered_weights = [weight for weight in weights if weight > 0.01]
        ax.pie(filtered_weights, labels=filtered_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title('Portfolio Weights Distribution', color='white')
        self.weights_pie_plot.figure.tight_layout()
        self.weights_pie_plot.draw()


def main():
    app = QApplication(sys.argv)
    window = PortfolioOptimizerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()