# nlp-classifier


### Running the scripts
I am not exactly sure how to set up the folder structure so that all the import statements can work correctly. Workaround: created a `.profile` file that appends my code to `PYTHONPATH`.

So, when running from the shell this profile needs to be loaded first:

```python
# source once, and this shell will have the updated path until it is closed.
source .profile
python src/mvp.py
```

The `.env` file is for running the code with visual studio code, its corresponding settings are in `.vscode/settings.json`.
