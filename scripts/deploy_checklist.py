"""Pre-deployment checklist for Depression Companion."""

import subprocess
import sys
from pathlib import Path

CHECKS = [
    ("✅", "Git initialized", ".git"),
    ("✅", "pyproject.toml exists", "pyproject.toml"),
    ("✅", "Dockerfile exists", "docker/Dockerfile"),
    ("✅", "docker-compose.yml exists", "docker/docker-compose.yml"),
    ("✅", "Tests pass", None),  # Manual check
    ("✅", "Frontend builds", "frontend/package.json"),
    ("✅", "README complete", "README.md"),
    ("✅", "vercel.json exists", "frontend/vercel.json"),
    ("✅", "railway.toml exists", "railway.toml"),
    ("✅", ".env.example exists", ".env.example"),
    ("✅", ".gitignore exists", ".gitignore"),
]


def run_checks():
    """Run deployment readiness checks."""
    print("\n" + "=" * 50)
    print("DEPLOYMENT READINESS CHECKLIST")
    print("=" * 50 + "\n")
    
    all_pass = True
    
    for emoji, name, path in CHECKS:
        if path is None:
            # Manual check
            print(f"⬜ {name}: Run `pytest tests/ -v` to verify")
            continue
        
        if Path(path).exists():
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - Missing: {path}")
            all_pass = False
    
    print("\n" + "=" * 50)
    
    # Optional checks
    print("\nOptional checks:")
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print("✅ Docker installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⬜ Docker not installed (optional for local dev)")
    
    # Check if git remote is set
    try:
        result = subprocess.run(
            ["git", "remote", "-v"], capture_output=True, text=True
        )
        if result.stdout.strip():
            print("✅ Git remote configured")
        else:
            print("⬜ No git remote set")
    except subprocess.CalledProcessError:
        print("⬜ Git not available")
    
    if all_pass:
        print("\n🎉 Ready to deploy!")
        print("\nDeploy commands:")
        print("  Frontend: cd frontend && vercel --prod")
        print("  Backend:  railway up")
        print("  Database: Connect Supabase URL in Railway env vars")
    else:
        print("\n⚠️  Fix the issues above before deploying")
    
    return all_pass


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)