# Contributing to Mater

[TOC]

## ✨ Workflow Overview

This project follows a **GitFlow** branching model on GitLab. Below is the standard process:

1. All tasks require an **issue** to be created on GitLab. This includes feature development, bug fixes, and documentation updates.
2. Assign the issue to a relevant **milestone** to indicate its purpose and timeline when possible.
3. Create a branch from the issue (from the `dev` branch for features or from the `main` branch for hotfixes).
4. Submit a **Merge Request (MR)** targeting the appropriate branch (`dev` for features, `main` for hotfixes).
5. Ensure your code passes all checks (e.g., linting, tests) and includes sufficient documentation.

## 🔧 Getting Started

### 🐛 Report an Issue

1. Check for [existing issues](https://gricad-gitlab.univ-grenoble-alpes.fr/isterre-dynamic-modeling/mater-project/mater/-/issues) and add a reaction or a comment to it.
2. If the issue does not exist, you can create a [new issue](https://gricad-gitlab.univ-grenoble-alpes.fr/isterre-dynamic-modeling/mater-project/mater/-/issues/new):
   - Add a meaningful title.
   - Add a description for the developers to understand the issue.
   - Assign a developer to the task, if applicable.
   - Assign the issue to the most appropriate **milestone** based on the task priority and timeline.
   - Use labels to categorize the issue (e.g., `bug`, `documentation`, `feature`).

### 🛠️ Contribute as a Developer

We welcome contributions from everyone ! Feel free to help on any issue.

#### 💁 First-Time Contributor

If you are new to this project or open source in general, don’t worry! We’ve outlined everything you need to get started. If you get stuck, feel free to ask for help by commenting on an issue or opening a discussion.

##### Set Up Your Environment

The recommended IDE is [Visual Studio Code](https://code.visualstudio.com/) (VSCode). Add the VSCode extension "Ruff".

You can configure your `settings.json` file to enable Ruff linting fixes and formatting when saving your script. Open `settings.json` by pressing `CTRL+SHIFT+P`, typing `settings`, and selecting `Preferences: Open User Settings (JSON)`.

Add the following to configure Ruff on save:

```json
{
  "git.autofetch": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

###### Linux

Install necessary dependencies:

```bash
sudo apt update
sudo apt install curl git
```

Install the package manager [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python 3.12 or higher:

```bash
uv python install 3.12
```

###### Windows

Install [Git](https://git-scm.com/).

Install the package manager [uv](https://docs.astral.sh/uv/):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install Python 3.12 or higher:

```bash
uv python install 3.12
```

Make sure Python is added to your system PATH during installation. If not, follow the [uv documentation](https://docs.astral.sh/uv/guides/install-python/#getting-started) to add it manually.

##### Clone the Project

Clone the project to your local machine:

```bash
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/isterre-dynamic-modeling/mater-project/mater.git
```

#### 🔄 Keep Your Branch Up-to-Date

Regularly update your branch with the latest changes from `dev` to avoid conflicts:

1. Fetch the latest changes for all branches:

```bash
git fetch --all
```

2. Rebase your branch on top of `dev`:

```bash
git rebase origin/dev
```

> **_TIPS_**
>
> Resolve conflicts (if any):
>
> - Git will pause and indicate conflicts. Open the conflicted files, resolve the issues, and stage the resolved changes:
>
> ```bash
> git add <file>
> ```
>
> - Continue the rebase after resolving conflicts:
>
> ```bash
> git rebase --continue
> ```

3. Push your rebased branch:

After rebasing, force push your branch to update the remote branch:

```bash
git push --force-with-lease
```

#### 🧪 Tests (Coming Soon)

Currently, this project does not have automated tests. However, we are actively working on introducing a comprehensive testing framework in future versions. Once implemented, contributors will be expected to write and run tests for their changes. Stay tuned for updates!

#### 📤 Submitting Your Work

1. Open a Merge Request:

   - Go to Merge Requests in the Code tab of the repository.
   - Click "New Merge Request" and select your branch as the source and `dev` as the target branch.
   - Provide a clear description of your changes and link any related issues (this is automatic if the branch was created from an existing issue).

   ##### Example Merge Request Description

   ```
   Title: feat(issue-123): Add seismic analysis feature

   Description:
   - Implements a new seismic analysis feature.
   - Adds `seismic_analysis.py` and updates documentation.
   - Addresses issue #123.

   Checklist:
   - [x] Code adheres to style guidelines (`ruff check`).
   - [x] Documentation updated.
   ```

2. Address Feedback:

   - Be prepared to revise your changes based on feedback from reviewers.

#### 🧹 Branch Cleanup

After your branch has been merged, delete it to keep the repository tidy:

```bash
git remote prune origin
git branch -D <YOUR_BRANCH_NAME>
```

## 📚 Online Documentation

**See the official [documentation](https://isterre-dynamic-modeling.gricad-pages.univ-grenoble-alpes.fr/mater-project/mater/contributing/index.html#contributing)**.

## 📜 Code of Conduct

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).
