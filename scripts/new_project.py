import os
import sys

def generate_project_structure(name):
    project_dir = f'projects/{name}'
    os.makedirs(project_dir, exist_ok=True)

    # Create standard directory structure
    directories = ['train', 'evaluate', 'predict', 'config']
    for directory in directories:
        os.makedirs(os.path.join(project_dir, directory), exist_ok=True)

    # Create a simple __main__.py file
    main_file_content = """
if __name__ == '__main__':
    print('Running the project!')
"""
    with open(os.path.join(project_dir, '__main__.py'), 'w') as main_file:
        main_file.write(main_file_content)

    # Create a uv-friendly pyproject.toml
    toml_content = """
[build-system]
requires = ['setuptools', 'wheel']
build-backend = 'setuptools.build_meta'
"""
    with open(os.path.join(project_dir, 'pyproject.toml'), 'w') as toml_file:
        toml_file.write(toml_content)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python new_project.py <project_name>')
        sys.exit(1)
    project_name = sys.argv[1]
    generate_project_structure(project_name)