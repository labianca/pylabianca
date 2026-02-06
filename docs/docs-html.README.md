To build and preview the documentation locally:

1. Install the documentation dependencies (Sphinx and the configured extensions). For example:

   ```bash
   pip install sphinx nbsphinx
   ```

   or install the full project environment using the provided `environment.yml`.

2. From the repository root, build the HTML output:

   ```bash
   sphinx-build -b html docs/source docs/build/html
   ```

   You can also run `make html` from inside the `docs/` directory.

3. Open `docs/build/html/index.html` in your browser to explore the rendered site.
