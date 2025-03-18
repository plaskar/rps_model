library(dplyr)
library(ggplot2)
library(muhaz)
library(lubridate)


## ---- 1. Loading Data ----
df_exp_quit <- read.csv('../data/df_exp_quit.csv') # exponential sim
df_log_quit <- read.csv('../data/df_log_quit.csv') # logistic sim


## ---- 2. Get Quit Hazard rate ----
# df_log100 <- df_log_quit %>% filter(prac)

hazard_log <- muhaz(times = df_log_quit$prac_final, delta = df_log_quit$quit, 
                    min.time = 0, max.time = 1000,
                    n.min.grid = 51, n.est.grid = 101) # make hazard function

#kphaz_log <- kphaz.fit(time = df_log_quit$prac_final, status=df_log_quit$quit, q=1,method="nelson")
#kphaz.plot(kphaz_log)

# CANT CALCULATE HAZARD RATE FOR EXP CASE SINCE NO QUITTERS!!
# hazard_exp <- muhaz(times = df_exp_quit$prac_final, delta = df_exp_quit$quit, 
#                  n.min.grid = 101, n.est.grid = 201) # make hazard function

## ---- 3. Estimate hazard rates ---- 
hazard_log_df <- data.frame('time' = hazard_log$est.grid, 
                     'haz_rate' = hazard_log$haz.est) 
write.csv(hazard_log_df, file='../data/sim_log_haz_rates.csv', row.names = FALSE)
# hazard_exp_df <- data.frame('time' = hazard_exp$est.grid, 'haz_rate' = hazard_exp$haz.est) 

## ---- 4. Plotting ---- 
### ---- 4.1 Logistic ---- 
log_haz_plot <- ggplot(hazard_log_df, aes(x=time, y=haz_rate)) +
  geom_line() +
  labs(title="Logistic Simulation", 
       x="Time", 
       y="Quitting Hazard Rate") + 
  theme_bw() + 
  theme(legend.position = c(0.8, 0.8),
        element_rect(fill='white', color='black'))  # position legend inside top-right

log_haz_plot 
ggsave('../plots/sim_log_quit_haz_rate.jpg', plot=log_haz_plot, dpi = 256) 
