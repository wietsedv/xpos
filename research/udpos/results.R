setwd("/Volumes/Projects/xling-benchmarks")

df <- read.csv('results.csv')
df$ldnd_norm = (df$ldnd - min(df$ldnd)) / (max(df$ldnd) - min(df$ldnd))

#library(lme4)
library(lmerTest)

m0 = lmer('score ~ family_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m0)
m1 = lmer('score ~ family_same + script_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m1)
m2 = lmer('score ~ family_same + script_same + pretrained_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m2)
m3 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m3)
m4 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + ldnd + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m4)
m5 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + ldnd + train_size + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m5)  # train_size not significant
m6 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + ldnd + sampling + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m6)  # sampling not significant
m7 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m7)
m8 = lmer('score ~ family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + lang_pred_family + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m8)
m9 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m9)
m10 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m10)
m11 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m11)
m12 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m12)
m13 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_train_script + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m13)  # lang_train_script not significant
m14 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_pred_pretrained + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m14)
m15 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_pred_pretrained + lang_train_pretrained + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m15)

anova(m0, m1, m2, m3, m4, m7, m8, m9, m10, m11, m12, m14, m15)

MuMIn::r.squaredGLMM(m15)
anova(m15)


m0 = lmer('score ~ family_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m1 = lmer('score ~ family_same + script_type_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m2 = lmer('score ~ family_same + script_type_same + script_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m3 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m4 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + sov_order_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m5 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m6 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + train_size + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)  # not train_size
m7 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + sampling + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)  # not sampling
m8 = lmer('score ~ family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m9 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m10 = lmer('score ~ lang_pred_family + lang_train_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)  # lang_train_family not significant
m11 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m12 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m13 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m14 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_train_script + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)  # lang_train_script not significant
m15 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_pred_pretrained + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m16 = lmer('score ~ lang_pred_family + family_same + script_type_same + script_same + pretrained_same + sov_order_same + ldnd + ie_same + sov_order_pred + sov_order_train + lang_pred_script + lang_pred_pretrained + lang_train_pretrained + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
anova(m0, m1, m2, m3, m4, m5, m8, m9, m11, m12, m13, m15, m16)

MuMIn::r.squaredGLMM(m16)
anova(m16)

m = lmer('score ~ family_same + ie_same + script_type_same + script_same + pretrained_same + sov_order_same + lang_train_pretrained + lang_pred_pretrained + lang_pred_family + lang_pred_script_type + lang_pred_script + ldnd + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE); summary(m)


m0 = lmer('score ~ family_same + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m1 = lmer('score ~ family_same + script_type_same + script_same + sov_order_same + lang_pred_pretrained + lang_train_pretrained + ldnd + (1|lang_pred_family) + (1|lang_pred) + (1|lang_train)', data=df, REML=FALSE)
m2 = lmer('score ~ family_same + script_type_same + script_same + sov_order_same + lang_pred_pretrained * lang_train_pretrained + ldnd + (1|lang_pred_family) + (1|lang_pred) + (1|lang_train)', data=df, REML=TRUE)
anova(m0, m1, m2)

mf1 = lmer(score ~ family_same + lang_train_pretrained * lang_pred_pretrained + ldnd_norm + script_same + script_type_same + sov_order_same + (1|lang_pred_family) + (1|lang_pred) + (1|lang_train) , data=df, REML=T)
mf2 = lmer(score ~ family_same + related_train_pretrained * related_pred_pretrained + ldnd_norm + script_same + script_type_same + sov_order_same + (1|lang_pred_family) + (1|lang_pred) + (1|lang_train) , data=df, REML=T)
mf3 = lmer(score ~ family_same + lang_train_pretrained * lang_pred_pretrained + ldnd_norm + script_same + script_type_same + sov_order_same + train_size_tiny + (1|lang_pred_family) + (1|lang_pred) + (1|lang_train) , data=df, REML=T)
anova(mf1, mf3)

summary(mf1)

MuMIn::r.squaredGLMM(mf1)
MuMIn::r.squaredGLMM(mf2)

summary(mf1)
summary(mf2)

