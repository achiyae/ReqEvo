# ReqEvo
ReqEvo provides a dataset of requirements evolution and a tool that helps to extand this dataset with further domains.

## Artifacts
The repo provides some kinds of artifacts:
1. JSON: requirements versions comparison in a json format, located in `outputs` directory
2. Final HTML reports: static HTML files of the requirements analysis, located in `final_reports\` directory
3. HTML reports: dynamic HTML files that can be used only while running the code and allow the user to give some feedback of the anaylsis, located in `reports\` directory

## How to Extand\Edit the Dataset?
1. Clone the repo.
2. Run `python main.py` and follow the steps in the terminal.
3. In order to add a new domain, type "N" and provide a git file URL.
4. In order to edit an already exsiting one, type "R" and enter the name of the domain you would like to review.
5. In both cases, an HTML editable report will appear. You can provide some feedback and submit it.
   There is a button for submitting the feedback during the process that allows some iterations of re-analysis, and another button to approve the analysis that generates a final HTML report when you finish.
