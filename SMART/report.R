library(knitr)
library(tidyverse)
library(lubridate)
library(survival)
library(survminer)
library(gtsummary)
library(ISwR)
library(KMsurv)

# load dataset 

load("~/health/SMART.RData")

# display data and basic summary
head(SMART)
summary(SMART)
# a lot of NA's on most quantitative data...

# correlaizoni
library(PerformanceAnalytics) 
chart.Correlation(SMART, histogram=TRUE, pch=19) 

# how many died?
table(SMART$EVENT==1)
#FALSE  TRUE 
#3413   460 

# M/F ratio?
table(SMART$SEX)
#1     2    (check assumption that 1==M, 2==F)
#2897  976 

## Load survival package
library(survival)

# add a survival object
SMART$SurvObj <- with(SMART, Surv(TEVENT, EVENT == 1))

## Kaplan-Meier estimator. The "log-log" confidence interval is preferred.
km.as.one <- survfit(SurvObj ~ 1, data = SMART)#, conf.type = "log-log")
plot(km.as.one)

# cox regression

broom::tidy(
  coxModel <- coxph(Surv(TEVENT, EVENT) ~ alcohol, data = SMART),
  exp = TRUE
) %>% 
  kable()


mv_fit <- coxph(Surv(TEVENT, EVENT) ~ alcohol, data = SMART)
cz <- cox.zph(mv_fit)
print(cz)
plot(cz)
