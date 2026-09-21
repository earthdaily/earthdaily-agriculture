"""AI context generation utilities shipped with the package.

The :mod:`earthdaily.agriculture.scripts.generate_ai_context` module produces
``CLAUDE.md``, ``agents.md``, and the project-local / personal ``/earthdaily-agriculture``
skill from introspecting the extractor classes registered in
``EXTRACTOR_REGISTRY``. The cookiecutter post-generation hook invokes it
via ``python -m earthdaily.agriculture.scripts.generate_ai_context --project-target …``.
"""
