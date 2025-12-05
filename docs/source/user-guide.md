# User Guide

## E-GRAFS — User Guide (Private Package via GitLab)

> **Status:** Private project. You must be granted access to the repository and receive a **Deploy Token** (username + token) from the maintainer to install the package.

---

### 1) Installation (via GitLab Package Registry + Deploy Token)

**Project ID:** `35690`

#### 1.1 Set credentials (recommended via environment variables)
On macOS/Linux:
```bash
export GL_DEPLOY_USER="<deploy-username>"
export GL_DEPLOY_TOKEN="<deploy-token>"
```

On Windows (PowerShell):
```powershell
$env:GL_DEPLOY_USER="<deploy-username>"
$env:GL_DEPLOY_TOKEN="<deploy-token>"
```

#### 1.2 Install with `pip`
```bash
pip install   --index-url "https://${GL_DEPLOY_USER}:${GL_DEPLOY_TOKEN}@gitlab.com/api/v4/projects/35690/packages/pypi/simple"   --extra-index-url "https://pypi.org/simple"   E-GRAFS
```

> Tip: to avoid typing credentials each time, you can add the following to your `~/.config/pip/pip.conf` (Linux), `~/Library/Application Support/pip/pip.conf` (macOS), or `%APPDATA%\pip\pip.ini` (Windows). Keep using env vars for secrets.
```
[global]
index-url = https://${GL_DEPLOY_USER}:${GL_DEPLOY_TOKEN}@gitlab.com/api/v4/projects/35690/packages/pypi/simple
extra-index-url = https://pypi.org/simple
```

#### 1.3 Verify installation
```bash
python -c "import grafs_e, sys; print('E-GRAFS version:', getattr(grafs_e, '__version__', 'unknown'))"
```

**Upgrading**:
```bash
pip install -U E-GRAFS
```

---

### 2) Prepare your input data

E-GRAFS expects common agricultural statistics and technical coefficients prepared per the model’s input schema.  
Please follow the rules defined in **`docs/source/input.md`** (input specification, files & fields, units, and validation hints).

---

### 3) Quick start: run the model

```python
# 1) Import
from grafs_e import DataLoader, NitrogenFlowModel

# 2) Load inputs (paths to your project config and your data file)
input_project_file_path = "path/to/project.yml"   # or .json/.toml as applicable
input_data_file_path    = "path/to/data.xlsx"     # or .csv/.parquet/.sqlite etc.

data = DataLoader(input_project_file_path, input_data_file_path)

# 3) Instantiate and run
territory = "MyRegion"
year = 2022
m = NitrogenFlowModel(data, territory, year)
```

---

### 4) Access model inputs & outputs

After the model is instantiated (`m`), you can retrieve key data objects:

- **Cultures (crops)**:  
  ```python
  m.df_cultures
  ```

- **Elevage (livestock/herds)**:  
  ```python
  m.df_elevage
  ```

- **Populations**:  
  ```python
  m.df_pop
  ```

- **Products**:  
  ```python
  m.df_prod
  ```

- **Optimization results (allocations)**:  
  ```python
  m.allocations_df
  ```

- **Diet deviations (defined vs. allocated diets)**:  
  ```python
  m.deviations_df
  ```

- **System-wide transition matrix**:  
  ```python
  m.get_transition_matrix()
  ```

> All DataFrames/objects are returned in canonical units and naming conventions used by the model. See the API reference for column details.

---

### 5) Troubleshooting

- **401 Unauthorized**: wrong or expired token/username; or missing scope. Ask the maintainer for a fresh **Deploy Token** with `read_package_registry`.
- **403 Forbidden**: your account is not a member of the private project/group. Ask the maintainer to add you.
- **404 Not Found**: the package name/version is not published to the project’s registry, or the index URL is incorrect.
- **SSL/Proxy issues**: configure corporate proxy/SSL as required by your environment.
- **Rotation**: tokens usually have an expiry; request a new Deploy Token ahead of time and update your env vars and/or `pip.conf`.

---

### 6) Security notes

- **Never commit** tokens to Git (use env vars or a local `pip.conf` not tracked by VCS).
- Prefer **short-lived** tokens and rotate them periodically.
- If you automate installs in CI, store credentials in **CI/CD masked variables** and reference them in the job environment.

---

### 7) Uninstall

```bash
pip uninstall E-GRAFS
```

---

### 8) What’s next?

- Read **`docs/source/overview.md`** to understand the physical/systemic principles of E-GRAFS.
- Use **`docs/source/user-guide.md`** alongside **`docs/source/input.md`** to validate your data and run first scenarios.
- Explore the API reference for advanced usage and reporting helpers.