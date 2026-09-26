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

You only need `ANTHROPIC_API_KEY` for commands that call Claude — `ask`,
`chat`, and `audio`. `ingest` and `search` work without it. See the next
section for how to get one.

**Check it worked:** open the file again and confirm your edits are saved
(Notepad shows no "unsaved changes" indicator).

### Getting an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com) and sign in
   (or create an account).
2. Go to **API keys** → **Create key**, give it a name (e.g. "NoteCast"),
   and copy the key it shows you — it starts with `sk-ant-`. You won't be
   able to see it again after you leave the page, so copy it now.
3. Paste it into your `.env` file after `ANTHROPIC_API_KEY=`, with no
   spaces or quotes, and save.
4. **Set a spending limit before you do anything else.** In the console,
   go to **Settings → Limits** (or **Billing**) and set a **monthly spending
   limit**. A low limit (a few dollars) is plenty while you're learning
   NoteCast — every command that spends money (`ask`/`chat` in `open` or
   `deep` mode, and `audio`) shows you an estimated cost and asks you to
   confirm before it sends anything, but a spending limit is still a good
   safety net in case of a mistake or a bug.

**Keep your key secret.** Anyone with your API key can spend your credits.
Never commit your real `.env` file (the `.gitignore` already blocks a
normal `git add`/`git commit` from picking it up), never paste your key into
a chat, an issue, or a screenshot, and if you ever think it's leaked,
revoke it in the console and create a new one.

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

## 7. Run ingest to parse and index your files

**NVIDIA/CUDA is not required.** Local search (embeddings) runs on the CPU
through an ONNX runtime, which is fast enough on a normal laptop — no GPU
driver or PyTorch install needed.

Once you've added files to a notebook's `sources/` folder (step 6), run:

```powershell
uv run notecast ingest comp4650-document-analysis
```

This reads your files (parses PDFs/slides/transcripts into text and splits
them into small searchable pieces) and indexes them, so later questions can
find the relevant piece instead of re-reading everything.

**First run downloads a search model (~70 MB, once)** into `.cache/models`
inside the project folder, so it needs internet access the first time only
— after that it's read from disk. Run `notecast` commands from the project
folder (`NoteCast/`) so it finds that cache.

**Check it worked:** the command prints a summary of files parsed/chunked,
with no errors.

## 8. Generate your first audio overview (optional)

Audio overviews use a local text-to-speech engine called Kokoro, run
through `kokoro-onnx` — no separate install needed for this step, and
**`ffmpeg` is not required**: NoteCast stitches the MP3 itself in Python.

```powershell
uv run notecast audio comp4650-document-analysis
```

**First run only:** this downloads the Kokoro voice model (~350 MB total,
model + voices) into `.cache/models/kokoro`. It needs internet the first
time only; after that it's read from disk, the same way the embedding model
in step 7 works. It'll also print an estimated cost and ask you to confirm
before it calls Claude — see [`docs/USAGE.md`](USAGE.md#5-audio-overviews)
for what the estimate means and what the pipeline does.

**Check it worked:** an MP3 (plus a script and transcript) appears under
`notebooks/comp4650-document-analysis/audio/`.

## 9. Try the web UI (optional)

NoteCast also has a local web UI, built with Streamlit, covering the same
chat and audio features as the command line:

```powershell
uv run streamlit run notecast/ui/app.py
```

This should open a new browser tab automatically (or print a `localhost`
address to open yourself). Close it with `Ctrl+C` in the terminal when
you're done. See [`docs/USAGE.md`](USAGE.md#6-web-ui) for what's in it.

## 10. Optional: install VS Code

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

**"`git`/`uv` is not recognized as the name of a command"**
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
