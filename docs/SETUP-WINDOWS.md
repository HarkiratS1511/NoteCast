# Windows setup guide

This is a step-by-step guide to getting NoteCast running on Windows, written for
someone who hasn't set up a Python dev environment before. Follow the steps in
order. After each one there's a line telling you how to check it actually worked
before you move on.

You'll be using **PowerShell**, which is the command-line app that comes built
into Windows (it's the blue/black window where you type commands instead of
clicking things). We'll also use **Windows Terminal** if you have it — it's a
nicer wrapper around PowerShell with tabs. Either works fine.

## 1. Open PowerShell

Press `Win`, type `PowerShell` or `Terminal`, and hit Enter. Everything below
gets typed into this window, one command at a time.

**Check it worked:** you should see a prompt like `PS C:\Users\you>` waiting
for input.

## 2. Install Git

Git is the tool that downloads (and later updates) the NoteCast source code
from GitHub, and tracks changes to it.

```powershell
winget install --id Git.Git -e
```

`winget` is Windows' built-in app installer — it's already on modern Windows
10/11. If it's missing, install "App Installer" from the Microsoft Store first.

After installing, **close and reopen your terminal** (installers change your
PATH, the list of folders Windows searches for commands, and the current
window doesn't see that update).

Then tell Git who you are — it stamps this on every change you make:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**Check it worked:**

```powershell
git --version
```

should print something like `git version 2.4x.x`.

## 3. Install uv

`uv` is the package manager NoteCast uses. A **package manager** installs and
tracks the external code (dependencies) a project needs — think of it as an
app store for Python libraries, plus a way to record exactly which versions
you're using. `uv` also downloads and manages the right version of **Python**
itself, so you don't need to install Python separately.

```powershell
winget install --id astral-sh.uv -e
```

If `winget` can't find it, use this alternative install script instead:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Restart your terminal** again after installing.

**Check it worked:**

```powershell
uv --version
```

should print something like `uv 0.5.x`.

## 4. Clone the repo and install dependencies

"Cloning" means downloading a full copy of the project (with its history) onto
your machine.

```powershell
git clone https://github.com/HarkiratS1511/NoteCast.git
cd NoteCast
```

Now install the project's dependencies. `uv sync` reads `pyproject.toml` (the
file that lists everything NoteCast needs), downloads a matching Python 3.11+
automatically if you don't already have one, creates a private **virtual
environment** in a `.venv` folder (an isolated set of installed packages just
for this project, so it can't clash with anything else on your machine), and
installs everything into it:

```powershell
uv sync
```

This can take a few minutes the first time.

**Check it worked:**

```powershell
uv run notecast --help
```

should print NoteCast's list of commands, not an error.

## 5. Create your `.env` file

`.env` is a small text file that holds settings and secrets (like API keys)
that should never be shared or uploaded to GitHub — it's listed in
`.gitignore` (a file that tells Git "never track this") for exactly that
reason.

```powershell
Copy-Item .env.example .env
```

Open it in Notepad (or VS Code, see step 9) and fill it in:

```powershell
notepad .env
```

You don't need `ANTHROPIC_API_KEY` yet — that's only needed once we build the
chat feature (a later phase). When you do need it, get a key from
[console.anthropic.com](https://console.anthropic.com) → **API keys** → create
a new key, paste it in after `ANTHROPIC_API_KEY=`, and save. **Never commit
your real `.env` file** — the `.gitignore` already blocks it, so a normal
`git add`/`git commit` won't pick it up.

**Check it worked:** open the file again and confirm your edits are saved
(Notepad shows no "unsaved changes" indicator).

## 6. Create a notebook and add your material

A "notebook" in NoteCast is a folder for one course's material.

```powershell
uv run notecast notebooks create comp4650-document-analysis
```

Then copy your course files (PDFs, slides, transcripts) into
`notebooks/comp4650-document-analysis/sources/` using File Explorer, or nest
them by week, e.g. `notebooks/comp4650-document-analysis/sources/week-01/`.

**Important:** everything under `notebooks/` is git-ignored on purpose — see
the [Privacy note in the README](../README.md#privacy). This repo is public,
and your course material is copyrighted, so it must never be committed.

**Check it worked:**

```powershell
uv run notecast notebooks list
```

should show `comp4650-document-analysis` in the list.

## 7. (Phase 2 onward) Check your NVIDIA GPU

Later phases (local embeddings, then text-to-speech) run much faster on an
NVIDIA GPU. You don't need to do anything with this yet — just confirm Windows
can see your GPU and its driver:

```powershell
nvidia-smi
```

**Check it worked:** it prints a table with your GPU's name, driver version,
and memory. If the command isn't found, install the latest driver from
[nvidia.com/drivers](https://www.nvidia.com/drivers) first.

The exact command to install a CUDA-enabled build of PyTorch (the library
that runs the embedding model on your GPU) will be added here in **Phase 2**,
once we know the versions we need — don't install it yet.

## 8. (Phase 6 onward) Install ffmpeg

`ffmpeg` is the tool NoteCast will use to stitch together the individual
lines of a generated audio overview into one MP3 file. Not needed until the
audio phase (Phase 6):

```powershell
winget install --id Gyan.FFmpeg -e
```

Restart your terminal after installing.

**Check it worked:**

```powershell
ffmpeg -version
```

should print a version number, not an error.

## 9. Optional: install VS Code

VS Code is a code editor. It's optional, but it makes reading and editing
files (including this repo, your `.env`, and any code) much easier than
Notepad.

```powershell
winget install --id Microsoft.VisualStudioCode -e
```

Open VS Code, go to the Extensions tab (the icon with four squares on the left
sidebar), and install the **Python** extension (by Microsoft).

**Check it worked:** open the NoteCast folder in VS Code
(`File → Open Folder…`) and confirm you can see and edit files like
`README.md`.

---

## Troubleshooting

**"`git`/`uv`/`ffmpeg` is not recognized as the name of a command"**
The install added itself to your PATH, but your open terminal window doesn't
know that yet. Close the terminal completely and open a new one. If it still
fails, sign out and back into Windows (or restart).

**"running scripts is disabled on this system" (execution policy error)**
This happens with the `irm ... | iex` install script, not with `winget`. Run
this once to allow locally-run scripts for your user only (it doesn't affect
system security elsewhere):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the install script.

**Errors mentioning long file paths / "path too long"**
Windows historically limited file paths to 260 characters. Enable long path
support once, in an **admin** PowerShell window:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart your computer afterwards for it to take effect.

**First `uv sync` is very slow, or antivirus pops up**
Antivirus software (including Windows Defender) scans every new file `uv`
downloads and extracts, which can make the first `uv sync` noticeably slower
than later ones. This is normal — let it finish. If it seems stuck for a very
long time, check your antivirus isn't blocking network access for `uv.exe`.

**Getting the latest changes to the project**
Whenever there are updates to pull down:

```powershell
git pull
uv sync
```

`git pull` fetches the latest code, and `uv sync` makes sure your installed
dependencies still match what the project needs.
