# To-do list
Tasks to be done, grouped by priority.

## Tasks
### High priority
- Fix MemoryError when running with --hp-sweep.
- Make `waterNet` useable with other datasets.

### Medium priority
- Update `README.md` to reflect changes.
- Include `nb_layers` (number of CNN layers) in the hyperparameter sweep.
- Select appropriate hyperparameter distributions at runtime. Currently, they are hard-coded.

### Low priority
- Set up multiprocessing usage in the hyperparameter sweep.
- Make the code self-contained as a module, that can be imported into other Python scripts. Currently, importing this project is buggy.
- Make the terminal output more readable.