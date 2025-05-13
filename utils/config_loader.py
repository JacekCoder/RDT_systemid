"""
=========================
Script description
=========================
This script is used to load the configuration from a YAML file. Copied from Sherwin's code.
"""

import yaml
import argparse

class ConfigLoader:
    def __init__(self, default_yaml_file=None):
        self.yaml_file = default_yaml_file
        self.config = self.load_yaml() if self.yaml_file else {}

    def load_yaml(self):
        """
        Load configuration from a YAML file.
        """
        try:
            with open(self.yaml_file, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Configuration file {self.yaml_file} not found.")
            exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            exit(1)

    def override_with_args(self):
        """
        Handle command-line arguments and override YAML configuration.
        """
        parser = argparse.ArgumentParser(description="Override configuration settings.")
        parser.add_argument("--arg_file", type=str, help="Path to the YAML configuration file")
        
        # Dynamically generate arguments based on the YAML structure
        self._add_arguments(parser, self.config, prefix="")

        # Parse command-line arguments
        args = parser.parse_args()

        # If --arg_file is provided, load it and merge configurations
        if args.arg_file:
            self.yaml_file = args.arg_file
            self.config = self.load_yaml()

        # Override YAML values with other command-line arguments
        self._update_config_with_args(args, self.config, prefix="")

    def _add_arguments(self, parser, config, prefix):
        """
        Recursively add arguments to the parser based on the YAML structure.
        """
        for key, value in config.items():
            arg_name = f"--{prefix}{key}" if prefix else f"--{key}"
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                self._add_arguments(parser, value, prefix=f"{key}.")
            else:
                # Add argument for non-dictionary values
                parser.add_argument(arg_name, type=type(value), help=f"Override {arg_name}")

    def _update_config_with_args(self, args, config, prefix):
        """
        Recursively update the configuration with parsed arguments.
        """
        for key, value in config.items():
            arg_name = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                self._update_config_with_args(args, value, prefix=f"{key}.")
            else:
                # Update the value if an override exists
                arg_value = getattr(args, arg_name.replace(".", "_"), None)
                if arg_value is not None:
                    config[key] = arg_value

    def get_config(self):
        """
        Return the complete configuration.
        """
        return self.config