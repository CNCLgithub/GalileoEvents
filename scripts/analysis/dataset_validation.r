library(tidyverse)
library(rjson)

th <- theme_classic()
theme_set(th)

table_mass <- 0.027

trial_data <- read_csv("trial_data.csv") %>%
  mutate(material = factor(appearance, levels=c('Iron', 'Brick', 'Wood')),
         mass_ratio = mass / table_mass,
         path = factor(path))



# first filter out conditions for readability

single_cond = trial_data %>%
  filter(cond == 0)

# plot distributions of mass

single_cond %>%
  ggplot(aes(x = log2(density), fill = congruent)) +
  geom_histogram(binwidth = 0.5) + 
  facet_grid(type + shape ~ material)

single_cond %>%
  ggplot(aes(x = log(mass_ratio), fill = congruent)) +
  geom_histogram() + 
  facet_grid(type + shape ~ material)


# plot position distributions

single_cond %>%
  ggplot(aes(x = init_pos_ramp)) +
  geom_histogram() + 
  facet_grid(type + shape ~ material)

# base group counts
single_cond %>%
  group_by(type, material, shape) %>%
  summarise(n())

# distribution of collision timings

single_cond %>%
  group_by(material, shape) %>%
  mutate(z_time = scale(time)) %>%
  ggplot(aes(x = init_pos_ramp, y = z_time, color = congruent)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_grid(type + shape ~ material)

trial_data %>%
  ggplot(aes(x = factor(cond), y = time)) +
  geom_violin() +
  # geom_dotplot(binaxis = "y", stackdir = "center") + 
  facet_grid(type + shape ~ material)
  
# Examining condition list

condlist <- fromJSON(file = "condlist.json")
conditions <- do.call(cbind.data.frame, condlist)
colnames(conditions) <- make.unique(rep(letters, length.out = ncol(conditions)), sep='')

conditions <- conditions %>%
  pivot_longer(cols = colnames(.), 
               names_to = "exp_cond", 
               values_to = "path")


by_cond <- conditions %>%
  group_by(exp_cond) %>%
  right_join(trial_data, by = "path")

View(by_cond %>%
  group_by(scene) %>%
  summarise(n()))
