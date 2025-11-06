# Publishing Guide for browser-service Package

This guide provides step-by-step instructions for publishing the browser-service package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. **API Tokens**: Generate API tokens for uploading packages:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - Save these tokens securely!

3. **Required Tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Step-by-Step Publishing Process

### 1. Prepare Your Package Repository

```bash
# Clone your new repository
git clone https://github.com/YOUR_USERNAME/browser-service.git
cd browser-service

# Copy the browser_service folder from the main project
# Copy all files from the packaging/ folder
```

### 2. Update Package Metadata

Edit the following files with your information:

**setup.py**:
- Update `author`, `author_email`
- Update `url` and `project_urls` with your GitHub repo
- Verify `version` matches `browser_service/__init__.py`

**pyproject.toml**:
- Update `authors` and `maintainers`
- Update all URLs in `[project.urls]`

**README_PACKAGE.md** → **README.md**:
- Update badge URLs
- Update repository links
- Add any additional documentation

### 3. Verify Package Structure

Your repository should look like:
```
browser-service/
├── browser_service/
│   ├── __init__.py (with __version__ = "1.0.0")
│   ├── config.py
│   ├── agent/
│   ├── api/
│   ├── browser/
│   ├── locators/
│   ├── prompts/
│   ├── tasks/
│   └── utils/
├── tests/
├── setup.py
├── pyproject.toml
├── requirements.txt
├── MANIFEST.in
├── README.md
├── LICENSE
├── CHANGELOG.md
└── .gitignore
```

### 4. Test Package Building

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# This creates:
# - dist/browser-service-1.0.0.tar.gz (source distribution)
# - dist/browser_service-1.0.0-py3-none-any.whl (wheel)
```

### 5. Check Package Integrity

```bash
# Check the distribution
twine check dist/*

# Should output: "PASSED" for all files
```

### 6. Test Upload to TestPyPI (Recommended)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# When prompted, use:
# Username: __token__
# Password: <your TestPyPI API token>

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    browser-service

# The --extra-index-url allows dependencies from regular PyPI
```

### 7. Upload to PyPI (Production)

```bash
# Upload to production PyPI
twine upload dist/*

# When prompted, use:
# Username: __token__
# Password: <your PyPI API token>
```

### 8. Configure ~/.pypirc (Optional - For Easier Uploads)

Create `~/.pypirc` (Linux/Mac) or `%USERPROFILE%\.pypirc` (Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Then you can upload with:
```bash
twine upload --repository testpypi dist/*  # For TestPyPI
twine upload dist/*                         # For PyPI
```

### 9. Verify Installation

```bash
# Install from PyPI
pip install browser-service

# Test import
python -c "from browser_service import config; print(config.__version__)"
```

## Updating the Package

### For New Versions:

1. **Update Version Numbers**:
   - `browser_service/__init__.py`: Update `__version__`
   - `setup.py`: Update `version`
   - `pyproject.toml`: Update `version`

2. **Update CHANGELOG.md**:
   - Document all changes

3. **Create Git Tag**:
   ```bash
   git tag -a v1.0.1 -m "Release version 1.0.1"
   git push origin v1.0.1
   ```

4. **Rebuild and Upload**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   twine check dist/*
   twine upload dist/*
   ```

## Common Issues and Solutions

### Issue: Package name already exists
**Solution**: Choose a different name. Check availability at https://pypi.org/

### Issue: Version already exists
**Solution**: You cannot re-upload the same version. Increment the version number.

### Issue: Missing dependencies during installation
**Solution**: Verify all dependencies are listed in `requirements.txt` and `setup.py`

### Issue: Import errors after installation
**Solution**: Check that `__init__.py` files exist in all subdirectories and properly expose modules

## GitHub Actions (Optional - Automated Publishing)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add your PyPI token as a GitHub secret named `PYPI_API_TOKEN`.

## Best Practices

1. **Always test on TestPyPI first**
2. **Use semantic versioning** (MAJOR.MINOR.PATCH)
3. **Keep CHANGELOG.md updated**
4. **Tag releases in Git**
5. **Document breaking changes clearly**
6. **Test installation in clean environment before publishing**
7. **Keep API tokens secure** (never commit them)

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## Need Help?

- PyPI Support: https://pypi.org/help/
- Python Packaging Discord: https://discord.gg/pypa
