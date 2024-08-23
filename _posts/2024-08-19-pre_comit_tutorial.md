---
title: "A Comprehensive Guide to Pre-Commit Tools in Python"
categories:
- Python
- Software Development
- Version Control

tags:
- Python
- Pre-commit
- Code Quality
- Git Hooks

author_profile: false
---

Ensuring code quality is a critical aspect of software development. One way to maintain high standards is by using pre-commit tools, which automatically check code before it is committed to a repository. This tutorial will guide you through the setup and usage of pre-commit tools in Python, helping you catch issues early in the development process.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [What is Pre-commit?](#what-is-pre-commit)
- [Installing Pre-commit](#installing-pre-commit)
- [Configuring Pre-commit Hooks](#configuring-pre-commit-hooks)
- [Commonly Used Pre-commit Hooks](#commonly-used-pre-commit-hooks)
  - [Running Pre-commit Locally](#running-pre-commit-locally)
  - [Integrating Pre-commit with CI/CD](#integrating-pre-commit-with-cicd)
- [Custom Hooks](#custom-hooks)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)

## What is Pre-commit?

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. These hooks are scripts that run automatically before a commit is made, ensuring that the code adheres to certain standards and is free from common errors. By using pre-commit hooks, developers can catch issues early, improving code quality and reducing the chances of introducing bugs.

## Installing Pre-commit

To get started with pre-commit, you'll need to install it. Pre-commit can be installed using pip, the Python package manager. Run the following command to install pre-commit:

```bash
pip install pre-commit
```

Once installed, you can verify the installation by checking the version:

```bash
pre-commit --version
```

## Configuring Pre-commit Hooks

Pre-commit hooks are configured using a .pre-commit-config.yaml file in the root directory of your repository. This file specifies the hooks you want to run and their configurations. Here is an example configuration file:

```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v3.4.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
```

In this example, three hooks are configured: `trailing-whitespace`, `end-of-file-fixer`, and `check-yaml`. These hooks will check for trailing whitespace, ensure files end with a newline, and validate YAML files, respectively.

## Commonly Used Pre-commit Hooks

There are many pre-commit hooks available for various purposes. Some commonly used hooks in Python projects include:

- **Flake8**: Checks for Python style guide enforcement.
- **Black**: Automatically formats Python code to follow the PEP 8 style guide.
- **Isort**: Sorts imports in Python files.
- **PyLint**: Analyzes code for potential errors and enforces a coding standard.
- **Mypy**: Performs static type checking.

To add these hooks to your configuration, update the `.pre-commit-config.yaml` file:

```yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v3.4.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml

- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
    - id: black

- repo: https://github.com/pre-commit/mirrors-flake8
  rev: v3.9.2
  hooks:
    - id: flake8

- repo: https://github.com/timothycrosley/isort
  rev: 5.9.3
  hooks:
    - id: isort

- repo: https://github.com/PyCQA/pylint
  rev: v2.11.1
  hooks:
    - id: pylint

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v0.910
  hooks:
    - id: mypy
```

### Running Pre-commit Locally

To use pre-commit hooks, you need to install the pre-commit hooks into your Git repository. Run the following command in the root directory of your repository:

```bash
pre-commit install
```

This command installs the pre-commit hooks, which will now run automatically before each commit. To manually run all hooks on all files, use:

```bash
pre-commit run --all-files
```

### Integrating Pre-commit with CI/CD

Integrating pre-commit with your CI/CD pipeline ensures that the code quality checks are enforced on every push and pull request. Here's an example of how to integrate pre-commit with GitHub Actions:

Create a .github/workflows/pre-commit.yml file in your repository with the following content:

```yaml
name: pre-commit

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: pip install pre-commit
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
```

This workflow runs the pre-commit hooks on every push and pull request, ensuring that all code changes meet the specified quality standards.

## Custom Hooks

Sometimes, you may need to create custom hooks tailored to your specific needs. Custom hooks can be written in any language and configured in the `.pre-commit-config.yaml` file. Here’s an example of a custom hook that checks for TODO comments:

Create a script file, `check_todo.sh`:


```bash
#!/bin/bash
if grep -rnw '.' -e 'TODO'; then
    echo "TODO found in the codebase. Please address it before committing."
    exit 1
fi
```

Make the script executable:

```bash
chmod +x check_todo.sh
```

Add the custom hook to your .pre-commit-config.yaml file:

```yaml
repos:
- repo: local
  hooks:
  - id: check-todo
    name: Check for TODO comments
    entry: ./check_todo.sh
    language: script
    files: \.py$
```

This custom hook will scan for TODO comments in Python files and prevent the commit if any are found.

## Best Practices

When using pre-commit hooks, consider the following best practices to ensure an effective workflow:

1. **Start Small**: Begin with a few essential hooks that address the most critical issues in your codebase. Gradually add more hooks as needed to avoid overwhelming the team with too many checks at once.
2. **Run Hooks Locally**: Encourage developers to run pre-commit hooks locally before pushing changes. This practice catches issues early and reduces the chances of failed CI builds.
3. **Customize Configuration**: Tailor the `.pre-commit-config.yaml` file to your project's specific needs. Disable or configure hooks to suit your coding standards and requirements.
4. **Keep Dependencies Updated**: Regularly update the hooks and their versions to benefit from improvements and bug fixes. Use the `pre-commit autoupdate` command to update all hooks in the configuration file.
5. **Monitor CI/CD Integration**: Ensure that your CI/CD pipeline consistently runs pre-commit hooks. Address any issues promptly to maintain code quality and prevent regressions.
6. **Document Hooks and Processes**: Provide clear documentation for the pre-commit hooks used in your project. Include instructions on installing, configuring, and running hooks, making it easier for new team members to get started.
7. **Leverage Pre-commit Plugins**: Explore and use pre-commit plugins and integrations with popular tools and services to enhance your workflow. These plugins can offer additional functionality and streamline the setup process.

## Conclusion

Pre-commit tools are invaluable for maintaining high code quality in Python projects. By automating code checks and enforcing standards before commits, these tools help catch issues early, reduce technical debt, and improve collaboration within development teams. With this guide, you should be well-equipped to set up and use pre-commit tools effectively, ensuring that your codebase remains clean, consistent, and error-free.

By following the steps outlined in this tutorial, you can integrate pre-commit hooks into your workflow, customize them to fit your needs, and leverage them to maintain a high standard of code quality in your Python projects.

For further reading and more advanced configurations, refer to the [official pre-commit documentation](https://pre-commit.com/) and explore the wide range of available hooks and plugins.

Happy coding!
