#  Retail Sales Analysis (SQL Project)

##  Project Overview

This project demonstrates end-to-end data analysis using SQL on a retail sales dataset. It includes database creation, data cleaning, exploration, and solving real-world business queries.

---

##  Database Setup

###  Create Database

```sql
CREATE DATABASE RetailSalesAnalysis_utf ;
USE RetailSalesAnalysis_utf;
```

###  Create Table

```sql
CREATE TABLE retail_sales(
transactions_id int primary key,
sale_date date,
sale_time time,
customer_id int null,
gender varchar(15),
age int null,
category varchar(20),
quantiy int null,
price_per_unit float null,
cogs float null,
total_sale float null);
```

---

##  Data Cleaning

###  Check for Null Values

```sql
SELECT* FROM retail_sales
WHERE age IS NULL OR quantiy IS NULL OR price_per_unit IS NULL OR cogs IS NULL OR total_sale IS NULL OR sale_time  IS NULL;
```

###  Delete Null Records

```sql
DELETE FROM retail_sales
WHERE age IS NULL OR quantiy IS NULL OR price_per_unit IS NULL OR cogs IS NULL OR total_sale IS NULL OR sale_time  IS NULL; 
```

---

##  Data Exploration

```sql
SELECT COUNT(category) FROM retail_sales;
```

---

##  Data Analysis & Queries

###  Q1: Sales on Specific Date

```sql
SELECT* FROM retail_sales
WHERE sale_date= "2022-11-05";
```

---

###  Q2: Clothing Transactions in November 2022

```sql
SELECT COUNT(*) FROM retail_sales
WHERE category ="clothing" AND quantiy>=4 AND sale_date BETWEEN "2022-11-01" AND "2022-11-30";
```

---

###  Q3: Total Sales by Category

```sql
SELECT category,SUM(total_sale) AS NET_SALE,COUNT(customer_id) AS CUSTOMER FROM retail_sales
GROUP BY category;
```

---

###  Q4: Category-wise Sales with Average Age

```sql
 SELECT category,SUM(total_sale) AS NET_SALE,COUNT(customer_id) AS CUSTOMER,AVG(age) AS AVG_AGE FROM retail_sales
GROUP BY category;
```

---

###  Q5: Transactions with Sales Greater Than 1000

```sql
SELECT* FROM retail_sales
WHERE total_sale>1000;
```

---

###  Q6: Transactions by Gender and Category

```sql
SELECT category,gender,COUNT(transactions_id) AS TOTAL FROM retail_sales
GROUP BY category,gender;
```

---

###  Q7: Best Selling Month Each Year

```sql
SELECT *
FROM (
    SELECT 
        YEAR(sale_date) AS year,
        MONTH(sale_date) AS month,
        AVG(total_sale) AS avg_sale,
        RANK() OVER (
            PARTITION BY YEAR(sale_date) 
            ORDER BY AVG(total_sale) DESC
        ) AS rank_in_year
    FROM retail_sales
    GROUP BY YEAR(sale_date), MONTH(sale_date)
) AS ranked_sales
WHERE rank_in_year = 1;
```

---

###  Q8: Top 5 Customers by Sales

```sql
SELECT customer_id, SUM(total_sale) AS sales FROM retail_sales
GROUP BY customer_id
ORDER BY sales DESC 
LIMIT 5;
```

---

###  Q9: Unique Customers per Category

```sql
SELECT category,COUNT(customer_id) 
FROM(
SELECT DISTINCT customer_id , category
FROM retail_sales) AS DIS
GROUP BY category;
```

---

###  Q10: Orders by Shift

```sql
SELECT 
    CASE 
        WHEN HOUR(sale_time) < 12 THEN 'Morning'
        WHEN HOUR(sale_time) BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS shift,
    COUNT(*) AS num_orders
FROM retail_sales
GROUP BY shift;
```

---

##  Key Insights

* Sales performance varies across different product categories
* Customer purchasing patterns differ based on demographics
* High-value transactions contribute significantly to total revenue
* Certain months perform better in terms of average sales
* Sales activity changes based on time of day

---

##  Tools Used

* SQL (MySQL)
* DBMS

---

##  Project Highlights

* Data Cleaning using SQL
* Aggregations and Grouping
* Window Functions (RANK)
* Business-Oriented Queries
* Real-world dataset analysis

---

##  Author

**Akash Badgoti**


