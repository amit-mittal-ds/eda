'''
Author   - Amit Mittal
Github   - https://github.com/amitmittal1005

Functions copied and are modified from below GitHub repository and few new functions added by myself 
Copied fromm: https://github.com/SharoonSaxena

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# custom function for easy and efficient analysis of numerical univariate

def UVA_numeric(data, var_group):
    """
      Univariate_Analysis_numeric
      takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

      Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
    """
    size = len(var_group)
    rows = int(size/3) + 1
    plt.figure(figsize = (15,4*rows), dpi = 500)
  
    #looping for each variable
    for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max()-data[i].min()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()
             # calculating points of standard deviation
        points = mean-st_dev, mean+st_dev

        #Plotting the variable with every information
        plt.subplot(rows,3,j+1)
        sns.kdeplot(data[i], shade=True) #hist = False,color = "pink", 
        sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
        sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
        sns.scatterplot([mean], [0], color = 'red', label = "mean")
        sns.scatterplot([median], [0], color = 'blue', label = "median")
        plt.xlabel('{}'.format(i), fontsize = 15)
        plt.ylabel('density')
        plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'\
                  .format((round(points[0],2),round(points[1],2)),round(kurt,2),round(skew,2), \
                          (round(mini,2),round(maxi,2),round(ran,2)),round(mean,2), round(median,2)),size=12)



# Custom function for easy visualisation of Categorical Variables
def UVA_category(data, var_group):

    '''
    Univariate_Analysis_categorical
    takes a group of variables (category) and plot/print all the value_counts and barplot.
    '''
    # setting figure_size
    size = len(var_group)
    rows=int(size/3) + 1
    plt.figure(figsize = (15,5*rows), dpi = 100)

    # for every variable
    for j,i in enumerate(var_group):
        norm_count = data[i].value_counts(normalize = True)
        n_uni = data[i].nunique()
        
        #Plotting the variable with every information
        plt.subplot(rows,3,j+1)
        sns.barplot(y=norm_count, x=norm_count.index , order = norm_count.index)
        plt.ylabel('fraction/percent', fontsize = 15)
        plt.xlabel('{}'.format(i), fontsize = 15)
        plt.xticks(rotation=90)
        plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni,norm_count))
        plt.tight_layout()

# custom function for easy outlier analysis



def UVA_outlier(data, var_group, include_outlier = True):
    '''
    Univariate_Analysis_outlier:
    takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
    Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

    data : dataframe from which to plot from\n
    var_group : {list} type Group of Continuous variables\n
    include_outlier : {bool} whether to include outliers or not, default = True\n
    '''

    size = len(var_group)
    plt.figure(figsize = (5*size,4), dpi = 100)
    for j,i in enumerate(var_group):

        # calculating descriptives of variable
        quant25 = data[i].quantile(0.25)
        quant75 = data[i].quantile(0.75)
        IQR = quant75 - quant25
        med = data[i].median()
        whis_low = quant25-(1.5*IQR)
        whis_high = quant75+(1.5*IQR)

        # Calculating Number of Outliers
        outlier_high = len(data[i][data[i]>whis_high])
        outlier_low = len(data[i][data[i]<whis_low])
        if include_outlier == True:
            #Plotting the variable with every information
            plt.subplot(1,size,j+1)
            sns.boxplot(data[i], orient="v")
            plt.ylabel('{}'.format(i))
            plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'\
                      .format(round(IQR,2),round(med,2),(round(quant25,2),round(quant75,2)),(outlier_low,outlier_high)))

        else:
            # replacing outliers with max/min whisker
            data2 = data[var_group][:]
            data2[i][data2[i]>whis_high] = whis_high+1
            data2[i][data2[i]<whis_low] = whis_low-1

            # plotting without outliers
            plt.subplot(1,size,j+1)
            sns.boxplot(data2[i], orient="v")
            plt.ylabel('{}'.format(i))
            plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'\
                      .format(round(IQR,2),round(med,2),(round(quant25,2),round(quant75,2)),(outlier_low,outlier_high)))



def BVA_categorical_plot(data, tar, cat,sig_level = 0.05):
    '''
    take data and two categorical variables,
    calculates the chi2 significance between the two variables 
    and prints the result with countplot & CrossTab
    '''
    #isolating the variables
    data = data[[cat,tar]][:]

    #forming a crosstab
    table = pd.crosstab(data[tar],data[cat],)
    f_obs = np.array([table.iloc[0][:].values,table.iloc[1][:].values])

    #performing chi2 test
    from scipy.stats import chi2_contingency
    chi, p, dof, expected = chi2_contingency(f_obs)
  
    #checking whether results are significant
    if p<sig_level:
        sig = True
    else:
        sig = False

    #plotting grouped plot
    plt.figure(figsize=(20,4))
    fig, axes = plt.subplots(1,2)
    ax1 = data.groupby(cat)[tar].value_counts(normalize=False).unstack().round(4)
    sns.countplot(x=cat, hue=tar, data=data, ax = axes[0])
    axes[0].set_xticklabels(data[cat].cat.categories,rotation=90)
    axes[0].title.set_text("p-value = {}\n Significance level: {}\n difference significant? = {}\n\n{}".format(round(p,8),\
                                                                                                        sig_level,sig,str(ax1)))
    
    #plotting percent stacked bar plot
    ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack().round(4)
    ax1.plot(kind='bar', stacked='True',title=str(ax1),ax= axes[1], figsize=(15,5))
    plt.xticks(rotation=90)
    plt.ylabel("Percentage")
    #int_level = data[cat].value_counts()
    


def Grouping_Data(data,var,val):
    '''
    It takes the data, varible, and the list of categories we want to group together in that variable
    It return the data frame in which the desired categories in the given column are named as 'Group 1' and rest all other 
    categories in that column named as 'Rest'
    '''
    rest = [x for x in data[var].cat.categories if x not in val]
    print("Group 1: {}".format(val))
    print("Rest: {}".format(rest))

    data2 = data.copy()
    data2[var] = data2[var].apply(lambda x: "Group 1" if x in val else "Rest")
    data2[var] = data2[var].astype("category")
    return data2



def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    '''
    takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
    '''
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval


def TwoSampT(X1, X2, sd1, sd2, n1, n2):
    '''
    takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sample T-Test
    '''
    from numpy import sqrt, abs, round
    from scipy.stats import t as t_dist
    ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
    t = (X1 - X2)/ovr_sd
    df = n1+n2-2
    pval = 2*(1 - t_dist.cdf(abs(t),df))
    return pval



def Bivariate_cont_cat(data, cont, cat, category,sig_level = 0.05,pos_class = "Positive Class",neg_class = "Negative Class"):
    #creating 2 samples
    x1 = data[cont][data[cat]==category][:]
    x2 = data[cont][~(data[cat]==category)][:]

    #calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.mean()

    #calculating p-values
    t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

    #checking whether results are significant
    if z_p_val < sig_level:
        sig = True
    else:
        sig = False
    
    #table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

    #plotting
    plt.figure(figsize = (15,4), dpi=140)
    
    #barplot
    plt.subplot(1,3,1)
    sns.barplot(['not {}'.format(category),str(category)], [m2, m1])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n Significance level: {} \ndifference significant? = {}\n {}'\
              .format(round(t_p_val,6), round(z_p_val,6),sig_level,sig,table))

    # category-wise distribution
    plt.subplot(1,3,2)
    sns.kdeplot(x1, shade= True, color='orange', label = pos_class)
    sns.kdeplot(x2, shade= False, color='blue', label = neg_class, linewidth = 1)
    plt.title('categorical distribution')
    # boxplot
    plt.subplot(1,3,3)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('categorical boxplot')



def Bivariate_Missing_Values(data2,missing,tar):
    '''
    This methods plots the stacked barplot for the Null values v/s not Null values for the variables w.r.t. target variable
    
    It takes input a dataframe, list of variables having missing values, target variable
    '''
    data = data2.copy()
    fig, axes = plt.subplots(1,len(missing))
    for i,val in enumerate(missing):
        data[val] = data[val].astype("object")
        data[val] = data[val].apply(lambda x: "Null" if pd.isnull(x) else "Not Null")
        
        ax1 = data.groupby(val)[tar].value_counts(normalize=True).unstack().round(4)
        ax1.plot(kind = "bar",stacked = True, figsize = (15,4), ax = axes[i])
        axes[i].set_ylabel("Percentage")
        axes[i].set_title("Null values: {}%\n {}".format(round(data[val].value_counts(normalize=True)[-1]*100,1),str(ax1)))
        

