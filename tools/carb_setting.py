import json
import argparse


def modify_json(file_path, log_level):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Modify the JSON log settings
    data['log']['outputStreamLevel'] = log_level
    data['log']['debugConsoleLevel'] = log_level

    # Save the modified JSON back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Modify JSON file log levels.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file')
    parser.add_argument('log_level',
                        type=str,
                        help='New log level (e.g., "fatal")')

    # Parse arguments
    args = parser.parse_args()

    # Call the function to modify the JSON
    modify_json(args.file_path, args.log_level)


if __name__ == "__main__":
    main()
