# DS project 1 - South Park

data <- read.csv("/Users/arianaelahi/Desktop/All-seasons.csv")
View(data)


library(ggplot2)
library(dplyr)

# 1. Number of lines per character (top 10)
data %>%
  count(Character, sort = TRUE) %>%
  top_n(10) %>%
  ggplot(aes(x = reorder(Character, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Characters by Number of Lines",
       x = "Character", y = "Number of Lines")

# 2. Number of lines per season
data$Season <- as.numeric(as.character(data$Season))

data %>%
  count(Season) %>%
  ggplot(aes(x = Season, y = n)) +
  geom_line(group = 1, color = "darkred") +
  geom_point(color = "darkred") +
  labs(title = "Total Number of Lines per Season",
       x = "Season", y = "Number of Lines")



filtered_data <- data %>%
  filter(Character %in% c("Cartman", "Stan", "Kyle", "Butters", "Randy",
                          "Mr. Garrison", "Chef", "Kenny", "Sharon", "Mr. Mackey"))
head(filtered_data)

write.csv(filtered_data, "filtered_data.csv", row.names = FALSE)
write.csv(filtered_data, "/Users/arianaelahi/Desktop/filtered_data.csv", row.names = FALSE)

