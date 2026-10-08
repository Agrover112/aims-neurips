# Security and private competition material

Report an accidentally committed credential to the repository owner privately. Revoke it before rewriting history.

Never commit API keys, Codabench tokens, private scoring downloads, hidden labels, or organizer-only material. Git ignore rules reduce mistakes but are not a security boundary. Review staged changes with `git diff --cached` before every push.

