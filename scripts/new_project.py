import os
import argparse


def create_standard_structure(project_id, project_name, slug, desc, python_version, force):
    # Validate project ID
    if not str(project_id).isdigit():
        raise ValueError('Project ID must be numeric')

    # Define project slug
    project_slug = slug or project_name.lower().replace(' ', '-')
    project_dir = f'projects/{project_slug}'

    # Prevent overwrite if force is not specified
    if os.path.exists(project_dir) and not force:
        raise FileExistsError(f'Project {project_slug} already exists. Use --force to overwrite.')

    # Create directory structure
    os.makedirs(f'{project_dir}/python/src/{project_slug}', exist_ok=True)
    os.makedirs(f'{project_dir}/data', exist_ok=True)
    os.makedirs(f'{project_dir}/reports', exist_ok=True)

    # Create required files
    with open(f'{project_dir}/README.md', 'w') as f:
        f.write(f'# {project_name}\n\n{desc or ""}')

    with open(f'{project_dir}/python/pyproject.toml', 'w') as f:
        f.write(f'[build-system]\nrequires = ["setuptools>=42", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{project_slug}"\ndescription = "{desc or ""}"\ndependencies = []\nscripts = {"train": "src/{project_slug}/train.py", "evaluate": "src/{project_slug}/evaluate.py", "predict": "src/{project_slug}/predict.py"}')

    for script in ['train', 'evaluate', 'predict']:  
        with open(f'{project_dir}/python/src/{project_slug}/{script}.py', 'w') as f:
            f.write(f'"""{script.capitalize()} script for {project_name}."""")

    with open(f'{project_dir}/python/src/{project_slug}/config.py', 'w') as f:
        f.write('"""Configuration settings for the project."""")

    with open(f'{project_dir}/python/src/{project_slug}/__init__.py', 'w') as f:
        f.write('"""Package init file."""")

    with open(f'{project_dir}/python/src/{project_slug}/__main__.py', 'w') as f:
        f.write('"""Executable script for the package."""")

    with open(f'{project_dir}/python/.python-version', 'w') as f:
        f.write(f'{python_version}')

    print("Project generated at: {}\nNext steps:\n- Create a virtual environment: uv venv\n- Install dependencies: uv sync\n- Run the project: uv run").format(project_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate a standardized project structure.')
    parser.add_argument('--id', type=int, required=True, help='Numeric project ID')
    parser.add_argument('--name', required=True, help='Project name')
    parser.add_argument('--slug', help='Optional project slug')
    parser.add_argument('--desc', help='Optional project description')
    parser.add_argument('--python', default='3.12', help='Python version')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing project')

    args = parser.parse_args()
    create_standard_structure(args.id, args.name, args.slug, args.desc, args.python, args.force)


if __name__ == '__main__':
    main()