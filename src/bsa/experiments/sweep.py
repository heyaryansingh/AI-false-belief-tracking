"""Sweep runner for parameter ablations and hyperparameter searches."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import product

from .runner import ExperimentRunner


class SweepRunner:
    """Runner for executing parameter sweeps and ablations."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ):
        """Initialize sweep runner.

        Args:
            config: Sweep configuration dictionary with keys:
                - parameter: Parameter name to sweep (e.g., 'num_particles')
                - values: List of values to test
                - base_config: Base experiment configuration
                - OR: parameters: Dict of {param_name: [values]} for grid search
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("results/sweeps")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[Dict[str, Any]] = []

    def run_sweep(self) -> Dict[str, Any]:
        """Run parameter sweep.

        Returns:
            Dictionary with sweep results summary
        """
        # Determine sweep type
        if "parameters" in self.config:
            # Grid search (multiple parameters)
            return self._run_grid_search()
        else:
            # Single parameter sweep
            return self._run_single_sweep()

    def _run_single_sweep(self) -> Dict[str, Any]:
        """Run single parameter sweep.

        Returns:
            Dictionary with sweep results summary
        """
        parameter = self.config.get("parameter")
        values = self.config.get("values", [])
        base_config = self.config.get("base_config", {})
        
        print(f"Running single parameter sweep: {parameter}")
        print(f"  Values: {values}")
        
        sweep_results = []
        
        for value in values:
            print(f"\n  Testing {parameter} = {value}")
            
            # Create modified config
            sweep_config = self._create_sweep_config(base_config, {parameter: value})
            
            # Run experiment
            runner = ExperimentRunner(sweep_config, output_dir=None)
            experiment_results = runner.run_experiment()
            
            # Collect results
            sweep_results.append({
                "parameter": parameter,
                "value": value,
                "experiment_results": experiment_results,
                "metrics": runner.results,
            })
        
        # Aggregate results
        aggregated = self._aggregate_sweep_results(sweep_results)
        
        # Save results
        self._save_sweep_results(sweep_results, aggregated, parameter)
        
        return {
            "parameter": parameter,
            "num_values": len(values),
            "aggregated_results": aggregated,
        }

    def _run_grid_search(self) -> Dict[str, Any]:
        """Run grid search over multiple parameters.

        Returns:
            Dictionary with grid search results summary
        """
        parameters = self.config.get("parameters", {})
        base_config = self.config.get("base_config", {})
        
        print(f"Running grid search over {len(parameters)} parameters")
        for param_name, values in parameters.items():
            print(f"  {param_name}: {values}")
        
        # Generate all combinations
        param_names = list(parameters.keys())
        param_value_lists = [parameters[name] for name in param_names]
        combinations = list(product(*param_value_lists))
        
        print(f"\n  Total combinations: {len(combinations)}")
        
        sweep_results = []
        
        for i, combination in enumerate(combinations):
            param_values = dict(zip(param_names, combination))
            print(f"\n  [{i+1}/{len(combinations)}] Testing: {param_values}")
            
            # Create modified config
            sweep_config = self._create_sweep_config(base_config, param_values)
            
            # Run experiment
            runner = ExperimentRunner(sweep_config, output_dir=None)
            experiment_results = runner.run_experiment()
            
            # Collect results
            sweep_results.append({
                "parameters": param_values,
                "experiment_results": experiment_results,
                "metrics": runner.results,
            })
        
        # Aggregate results
        aggregated = self._aggregate_grid_search_results(sweep_results, param_names)
        
        # Save results
        self._save_sweep_results(sweep_results, aggregated, "grid_search")
        
        return {
            "parameters": param_names,
            "num_combinations": len(combinations),
            "aggregated_results": aggregated,
        }

    def _create_sweep_config(
        self,
        base_config: Dict[str, Any],
        param_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create experiment config with parameter values set.

        Args:
            base_config: Base experiment configuration
            param_values: Dictionary of parameter names to values

        Returns:
            Modified configuration dictionary
        """
        import copy
        config = copy.deepcopy(base_config)
        
        # Set parameter values (handle nested paths like "model.belief_pf.particle_filter.num_particles")
        for param_name, value in param_values.items():
            if "." in param_name:
                # Nested parameter
                parts = param_name.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Top-level parameter
                config[param_name] = value
        
        return config

    def _aggregate_sweep_results(
        self,
        sweep_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate results across parameter values.

        Args:
            sweep_results: List of results for each parameter value

        Returns:
            Aggregated results dictionary
        """
        # Collect all metrics
        all_metrics = []
        for result in sweep_results:
            for metric_row in result["metrics"]:
                metric_row_copy = metric_row.copy()
                metric_row_copy["parameter_value"] = result["value"]
                all_metrics.append(metric_row_copy)
        
        if not all_metrics:
            return {}
        
        # Convert to DataFrame for aggregation
        df = pd.DataFrame(all_metrics)
        
        # Aggregate by parameter value
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        aggregated = {}
        
        for value in df["parameter_value"].unique():
            value_df = df[df["parameter_value"] == value]
            aggregated[str(value)] = {
                col: {
                    "mean": float(value_df[col].mean()) if col in numeric_cols else None,
                    "std": float(value_df[col].std()) if col in numeric_cols else None,
                    "min": float(value_df[col].min()) if col in numeric_cols else None,
                    "max": float(value_df[col].max()) if col in numeric_cols else None,
                }
                for col in numeric_cols
            }
        
        return aggregated

    def _aggregate_grid_search_results(
        self,
        sweep_results: List[Dict[str, Any]],
        param_names: List[str],
    ) -> Dict[str, Any]:
        """Aggregate results for grid search.

        Args:
            sweep_results: List of results for each parameter combination
            param_names: List of parameter names

        Returns:
            Aggregated results dictionary
        """
        # Collect all metrics
        all_metrics = []
        for result in sweep_results:
            for metric_row in result["metrics"]:
                metric_row_copy = metric_row.copy()
                for param_name, value in result["parameters"].items():
                    metric_row_copy[f"param_{param_name}"] = value
                all_metrics.append(metric_row_copy)
        
        if not all_metrics:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Aggregate by parameter combination
        grouped = df.groupby([f"param_{name}" for name in param_names])
        
        aggregated = {}
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        
        for name, group in grouped:
            aggregated[str(name)] = {
                col: {
                    "mean": float(group[col].mean()) if col in numeric_cols else None,
                    "std": float(group[col].std()) if col in numeric_cols else None,
                    "min": float(group[col].min()) if col in numeric_cols else None,
                    "max": float(group[col].max()) if col in numeric_cols else None,
                }
                for col in numeric_cols
            }
        
        return aggregated

    def _save_sweep_results(
        self,
        sweep_results: List[Dict[str, Any]],
        aggregated: Dict[str, Any],
        sweep_name: str,
    ) -> None:
        """Save sweep results to files.

        Args:
            sweep_results: List of results for each parameter value/combination
            aggregated: Aggregated results dictionary
            sweep_name: Name of the sweep
        """
        # Save detailed results as Parquet
        all_metrics = []
        for result in sweep_results:
            for metric_row in result["metrics"]:
                all_metrics.append(metric_row)
        
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            parquet_path = self.output_dir / f"{sweep_name}_results.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path, compression="snappy")
            print(f"\nDetailed results saved to: {parquet_path}")
        
        # Save aggregated results as JSON
        manifest = {
            "sweep_name": sweep_name,
            "config": self.config,
            "aggregated_results": aggregated,
        }
        
        manifest_path = self.output_dir / f"{sweep_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Aggregated results saved to: {manifest_path}")
