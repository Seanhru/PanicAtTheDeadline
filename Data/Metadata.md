## Summary of Data  
The data is a `.csv` file containing information regarding the **season**, **episode number**, **character**, and a **line** of dialogue from *South Park*.  
- Data types:  
  - `Season` and `Episode` → numerical values  
  - `Character` and `Line` → text strings  

The dataset can be downloaded from the team’s shared GitHub repository:  
🔗 [PanicAtTheDeadline Repository](https://github.com/Seanhru/PanicAtTheDeadline) [1]  

There are two versions of the dataset:  
- **Original dataset** → `All-seasons.csv`  
- **Cleaned dataset (ready to use for modeling)** → `cleaned_dataset.csv`  

---

## Provenance  
The dataset contains seasons, episodes, characters, and dialogue lines from the television show *South Park*, created by **Trey Parker** and **Matt Stone**.  
- While the majority of the show’s episodes were written by Parker and Stone, additional contributions from developers, directors, and guest writers are documented on the [South Park IMDb page][2].  
- The **original dataset** was compiled by **Bob Adams** from *SouthPark.Fandom.com*, covering seasons 1–18, and is publicly available on GitHub: [SouthParkData Repository](https://github.com/BobAdamsEE/SouthParkData) [3][4].  
- The source data originated as an `.html` file and was reformatted to `.csv` by Adams, who also corrected typos.  

On **September 12, 2025**, the team downloaded the dataset from Adams’ repository and transferred it to their own repository (linked in the summary section). Since then, the team has made modifications to the dataset. All subsequent changes and related work are documented in the team’s repository.  

---

## License  
The data is licensed under **Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** in the Creative Commons.  

---

## Data Dictionary  

| Column    | Description                                                        | Data Example / Range                                                                 |
|-----------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `Season`  | Season number of a particular episode of *South Park* (numeric).   | Any number **1–18**                                                                  |
| `Episode` | Episode number within the given season (numeric).                  | Any number **1–17**                                                                  |
| `Character` | The name of the character who delivers the line (string).        | *Cartman, Stan, Kyle, Butters, Randy, Mr. Garrison, Chef, Kenny, Sharon, Mr. Mackey* |
| `Line`    | The dialogue line spoken by the character (string).                | *"You guys, you guys! Chef is going away."*                                          |

---

## Ethical Statement  
All *South Park* characters, themes, and dialogue are the property of **Trey Parker** and **Matt Stone**. While the show may contain commentary or humor that is inappropriate for certain educational settings, the team focuses on analyzing **patterns in the data** to develop a functional machine learning model.  

---

## Exploratory Plots
![Top 10 Characters by Number of Lines](Output/Top_10_Characters_by_Number_of_Lines.png)
![Total Number of Lines per Season](Output/Total_Number_of_Lines_per_Season.png)

---

### References  
[1] PanicAtTheDeadline GitHub Repository: https://github.com/Seanhru/PanicAtTheDeadline  
[2] IMDb South Park Page: https://www.imdb.com/title/tt0121955/  
[3] Bob Adams’ SouthParkData GitHub Repository: https://github.com/BobAdamsEE/SouthParkData  
[4] SouthPark.Fandom.com: https://southpark.fandom.com  
