
-- 1. List all tables in the database
SELECT name FROM sqlite_master WHERE type='table';
-- 2. List all columns in the primary table
PRAGMA table_info(primary_data);

-- 3. Count total rows in the primary table
SELECT COUNT(*) AS total_rows FROM primary_data;

-- 4. Preview first 10 rows from the primary table
SELECT * FROM primary_data LIMIT 10;

-- 5. Get all column names and types dynamically
SELECT cid as column_index, 
       name as column_name, 
       type as data_type,
       [notnull] as not_null,
       dflt_value as default_value,
       pk as primary_key
FROM pragma_table_info('primary_data')
ORDER BY cid;

-- 6. Generate SQL to get distinct values for each column
SELECT 'SELECT DISTINCT [' || name || '] FROM primary_data;' as sql_query
FROM pragma_table_info('primary_data')
ORDER BY cid;

-- 7. Generate SQL to get min/max for numeric columns
SELECT 'SELECT MIN([' || name || ']) as min_' || name || 
       ', MAX([' || name || ']) as max_' || name || 
       ' FROM primary_data;' as sql_query
FROM pragma_table_info('primary_data')
WHERE type IN ('INTEGER', 'REAL', 'NUMERIC')
ORDER BY cid;

-- 8. Generate SQL to count NULLs for each column
SELECT 'SELECT COUNT(*) as null_count_' || name || 
       ' FROM primary_data WHERE [' || name || '] IS NULL;' as sql_query
FROM pragma_table_info('primary_data')
ORDER BY cid;

-- 9. Generate comprehensive column analysis for all columns
SELECT 'SELECT ''' || name || ''' as column_name, ' ||
       'COUNT(*) as total_rows, ' ||
       'COUNT([' || name || ']) as non_null_rows, ' ||
       'COUNT(*) - COUNT([' || name || ']) as null_rows, ' ||
       'ROUND(100.0 * COUNT([' || name || ']) / COUNT(*), 2) as completeness_pct' ||
       CASE 
           WHEN type IN ('INTEGER', 'REAL', 'NUMERIC') THEN 
               ', MIN([' || name || ']) as min_value, MAX([' || name || ']) as max_value'
           ELSE ''
       END ||
       ' FROM primary_data;' as sql_query
FROM pragma_table_info('primary_data')
ORDER BY cid;

-- 10. Get Datum information specifically (since you asked about it)
SELECT 'Datum Analysis:' as analysis_type;
SELECT DISTINCT [Datum], COUNT(*) as record_count 
FROM primary_data 
WHERE [Datum] IS NOT NULL
GROUP BY [Datum]
ORDER BY record_count DESC;

-- 11. Generate geographic analysis for coordinate columns
SELECT 'Geographic Analysis SQL:' as query_type;
SELECT 'SELECT ''' || name || ''' as coordinate_type, ' ||
       'MIN([' || name || ']) as min_coord, ' ||
       'MAX([' || name || ']) as max_coord, ' ||
       'AVG([' || name || ']) as avg_coord, ' ||
       'COUNT(DISTINCT [' || name || ']) as unique_values ' ||
       'FROM primary_data WHERE [' || name || '] IS NOT NULL;' as sql_query
FROM pragma_table_info('primary_data')
WHERE LOWER(name) LIKE '%lat%' OR LOWER(name) LIKE '%lon%' OR LOWER(name) LIKE '%coord%'
ORDER BY cid;

-- 12. Generate temporal analysis for date/time columns
SELECT 'Temporal Analysis SQL:' as query_type;
SELECT 'SELECT ''' || name || ''' as date_column, ' ||
       'MIN([' || name || ']) as earliest_date, ' ||
       'MAX([' || name || ']) as latest_date, ' ||
       'COUNT(DISTINCT [' || name || ']) as unique_dates ' ||
       'FROM primary_data WHERE [' || name || '] IS NOT NULL;' as sql_query
FROM pragma_table_info('primary_data')
WHERE LOWER(name) LIKE '%date%' OR LOWER(name) LIKE '%time%'
ORDER BY cid;

-- 13. Generate code analysis for categorical columns
SELECT 'Categorical Analysis SQL:' as query_type;
SELECT 'SELECT ''' || name || ''' as column_name, [' || name || '] as value, COUNT(*) as frequency ' ||
       'FROM primary_data WHERE [' || name || '] IS NOT NULL ' ||
       'GROUP BY [' || name || '] ORDER BY frequency DESC LIMIT 10;' as sql_query
FROM pragma_table_info('primary_data')
WHERE LOWER(name) LIKE '%code%' OR LOWER(name) LIKE '%name%' OR 
      LOWER(name) LIKE '%state%' OR LOWER(name) LIKE '%county%'
ORDER BY cid;

-- 14. Generate measurement analysis for value columns
SELECT 'Measurement Analysis SQL:' as query_type;
SELECT 'SELECT ''' || name || ''' as measurement_type, ' ||
       'COUNT(*) as total_measurements, ' ||
       'AVG([' || name || ']) as mean_value, ' ||
       'MIN([' || name || ']) as min_value, ' ||
       'MAX([' || name || ']) as max_value, ' ||
       'ROUND(AVG([' || name || ']) - MIN([' || name || ']), 2) as range_value ' ||
       'FROM primary_data WHERE [' || name || '] IS NOT NULL;' as sql_query
FROM pragma_table_info('primary_data')
WHERE LOWER(name) LIKE '%value%' OR LOWER(name) LIKE '%mean%' OR 
       LOWER(name) LIKE '%max%' OR LOWER(name) LIKE '%count%' OR LOWER(name) LIKE '%aqi%'
ORDER BY cid;

-- 15. EXECUTABLE ANALYSIS QUERIES (Ready to run - these produce actual data!)

-- Top 10 states by number of measurements
SELECT [State Name], COUNT(*) as measurement_count
FROM primary_data 
GROUP BY [State Name] 
ORDER BY measurement_count DESC 
LIMIT 10;

-- PM2.5 statistics by state
SELECT [State Name], 
       COUNT(*) as total_measurements,
       AVG([Sample Measurement]) as avg_pm25,
       MIN([Sample Measurement]) as min_pm25,
       MAX([Sample Measurement]) as max_pm25,
       ROUND(AVG([Sample Measurement]), 2) as rounded_avg
FROM primary_data 
GROUP BY [State Name] 
ORDER BY avg_pm25 DESC;

-- Monthly PM2.5 trends (2024)
SELECT substr([Date GMT], 1, 7) as year_month,
       COUNT(*) as measurements,
       AVG([Sample Measurement]) as avg_pm25,
       MAX([Sample Measurement]) as max_pm25
FROM primary_data 
GROUP BY year_month 
ORDER BY year_month;

-- Highest pollution episodes (PM2.5 > 50)
SELECT [State Name], [County Name], [Date GMT], [Time GMT], [Sample Measurement]
FROM primary_data 
WHERE [Sample Measurement] > 50
ORDER BY [Sample Measurement] DESC 
LIMIT 20;

-- Data completeness summary
SELECT 'Total Records' as metric, COUNT(*) as value FROM primary_data
UNION ALL
SELECT 'Records with PM2.5', COUNT(*) FROM primary_data WHERE [Sample Measurement] IS NOT NULL
UNION ALL
SELECT 'Unique States', COUNT(DISTINCT [State Name]) FROM primary_data
UNION ALL
SELECT 'Unique Counties', COUNT(DISTINCT [State Name] || ' - ' || [County Name]) FROM primary_data
UNION ALL
SELECT 'Date Range Days', julianday(MAX([Date GMT])) - julianday(MIN([Date GMT])) + 1 FROM primary_data;