# To-do list
Tasks to be done, grouped by priority.

## Tasks
### High priority
- Fix MemoryError when running with --hp-sweep.
- Make `waterNet` useable with other datasets.

### Medium priority
- Include `nb_layers` (number of CNN layers) in the hyperparameter sweep.
- Select appropriate hyperparameter distributions at runtime. Currently, they are hard-coded.

### Low priority
- Make the code self-contained as a module, that can be imported into other Python scripts. Currently, importing this project is buggy.
- Make the terminal output more readable.