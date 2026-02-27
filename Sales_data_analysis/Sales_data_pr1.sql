CREATE DATABASE RetailSalesAnalysis_utf ;
USE RetailSalesAnalysis_utf;
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

-- data cleaning
SELECT* FROM retail_sales
WHERE age IS NULL OR quantiy IS NULL OR price_per_unit IS NULL OR cogs IS NULL OR total_sale IS NULL OR sale_time  IS NULL;


DELETE FROM retail_sales
WHERE age IS NULL OR quantiy IS NULL OR price_per_unit IS NULL OR cogs IS NULL OR total_sale IS NULL OR sale_time  IS NULL; 

-- data exploration
SELECT COUNT(category) FROM retail_sales;

-- data analysis

-- My Analysis & Findings
-- Q.1 Write a SQL query to retrieve all columns for sales made on '2022-11-05
-- Q.2 Write a SQL query to retrieve all transactions where the category is 'Clothing' and the quantity sold is more than 10 in the month of Nov-2022
-- Q.3 Write a SQL query to calculate the total sales (total_sale) for each category.
-- Q.4 Write a SQL query to find the average age of customers who purchased items from the 'Beauty' category.
-- Q.5 Write a SQL query to find all transactions where the total_sale is greater than 1000.
-- Q.6 Write a SQL query to find the total number of transactions (transaction_id) made by each gender in each category.
-- Q.7 Write a SQL query to calculate the average sale for each month. Find out best selling month in each year
-- Q.8 Write a SQL query to find the top 5 customers based on the highest total sales 
-- Q.9 Write a SQL query to find the number of unique customers who purchased items from each category.
-- Q.10 Write a SQL query to create each shift and number of orders (Example Morning <=12, Afternoon Between 12 & 17, Evening >17)


SELECT* FROM retail_sales
WHERE sale_date= "2022-11-05";


SELECT COUNT(*) FROM retail_sales
WHERE category ="clothing" AND quantiy>=4 AND sale_date BETWEEN "2022-11-01" AND "2022-11-30";


SELECT category,SUM(total_sale) AS NET_SALE,COUNT(customer_id) AS CUSTOMER FROM retail_sales
GROUP BY category;


 SELECT category,SUM(total_sale) AS NET_SALE,COUNT(customer_id) AS CUSTOMER,AVG(age) AS AVG_AGE FROM retail_sales
GROUP BY category;


SELECT* FROM retail_sales
WHERE total_sale>1000;


SELECT category,gender,COUNT(transactions_id) AS TOTAL FROM retail_sales
GROUP BY category,gender;


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


SELECT customer_id, SUM(total_sale) AS sales FROM retail_sales
GROUP BY customer_id
ORDER BY sales DESC 
LIMIT 5;


SELECT category,COUNT(customer_id) 
FROM(
SELECT DISTINCT customer_id , category
FROM retail_sales) AS DIS
GROUP BY category;


SELECT 
    CASE 
        WHEN HOUR(sale_time) < 12 THEN 'Morning'
        WHEN HOUR(sale_time) BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS shift,
    COUNT(*) AS num_orders
FROM retail_sales
GROUP BY shift;




