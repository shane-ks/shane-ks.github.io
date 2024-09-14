# Querying
**NOTE:** To format outputs as an ASCII table, execute the command `.mode table`. 
### SELECT
This command selects, or pulls, information from the table.
```sql
SELECT * FROM "table";
SELECT "column" FROM "table";
SELECT "column1", "column2" FROM "table";
```
### LIMIT
This command limits the number of outputs.
```sql
SELECT "column" from "table" LIMIT 10;
```
### WHERE
This command is to pull information from the database that satisfies some condition. Note that single-quotes are used for the values if you are referring to a string. Additionally, the operators `!=` and `<>` are both the not-equal operator; they perform the same function.
```sql
SELECT "column1", "column2" FROM "table" WHERE "column3" = 2024;
SELECT "column1", "column2" FROM "table" WHERE "column3" = 'entry' LIMIT 10;
SELECT "column1", "column2" FROM "table" WHERE "column3" != 2024 LIMIT 10;
SELECT "column1", "column2" FROM "table" WHERE "column3" <> 2024 LIMIT 10;
```
### Logical Keywords
```sql
SELECT "column1", "column2" FROM "table" WHERE NOT "column3" = 2024;
SELECT "column1" FROM "table" WHERE "column3" = 2024 AND "column4" != 'string';
SELECT "column1" FROM "table" WHERE "column3" = 2024 OR "column4" != 'string';
SELECT "column1" FROM "table" 
	WHERE ("column3" = 2024 OR "column4" != 'string') 
	AND "column1" != 2;
```
### NULL
We can use `NULL` to find which data is missing.
```sql
SELECT "column1" FROM "table" WHERE "column2" IS NULL;
SELECT "column1" FROM "table" WHERE "column2" IS NOT NULL;
```
### Operators
```txt
% sign can match any character around a string that I give it 
	'Godzilla Minus One' matches with '%zilla%' or '%Godzilla%'
	'The Notebook' matches with 'The %b__k%'
	'The Poddleville Case' matches with 'The %Case%' but NOT with 'The %Case_%' 
_ can match any single character that I pass in with my string
	'Castleblanca' or 'XastleTlanc@' matches with '_astle_lanc_'
```
### LIKE
This keyword is used to roughly match some string in your table. Note that `LIKE` is case-insensitive.
```sql
SELECT "title" FROM "movies" WHERE "title" LIKE '%shawshank%';
SELECT "title" FROM "movies" WHERE "title" LIKE 'The %';
SELECT "title" FROM "movies" WHERE "title" LIKE 'The %book%';
SELECT "title" FROM "movies" WHERE "title" LIKE 'The %b__k%';
SELECT "title" from "movies" WHERE "title" LIKE '_astle_lanc_';
```
### Range Conditions
The operators that we can use are `>`, `<`, `>=`, and `<=`. We can also use `BETWEEN`, which works on dates as well.
```sql
SELECT "column1", "column2" FROM "table" 
	WHERE "column3" >= 2019 AND "column3" <= 2024;
SELECT "column1", "column2" FROM "table"
	WHERE "column3" BETWEEN 2019 AND 2024;
SELECT "column1", "column2" FROM "table"
	WHERE ("column3" BETWEEN 2019 AND 2024) AND "column4" > 5 LIMIT 10;
SELECT "column1" FROM "table"
	WHERE "column3" BETWEEN '2018-01-01' AND '2020-01-01';
```
### ORDER BY
The default ordering used by `ORDER BY` is smallest to largest. We can specify this ordering by using `ASC` and `DESC`. We can also order by multiple columns. For strings, `ORDER BY` sorts the strings alphabetically or reverse alphabetical order.
```sql
SELECT "column1" from "table" ORDER BY "column2" LIMIT 10;
SELECT "column1" from "table" ORDER BY "column2" DESC LIMIT 10;
SELECT "column1" from "table" ORDER BY "column2" DESC, "column3" DESC LIMIT 10;
```
### Aggregate Functions
The aggregate functions in SQL are `AVG`, `ROUND`, `MAX`, `MIN`, `SUM`, `COUNT`, and `DISTINCT`. Note that the `COUNT` aggregate function does not count `NULL` values. The functions `MIN` and `MAX` applied to strings return the earliest alphabetically ordered string and the largest, respectively. 
```sql
SELECT AVG("column1") FROM "table";
SELECT ROUND(AVG("column1"), 3) FROM "table"; 
SELECT ROUND(AVG("column1"), 3) AS "Average of Column1" FROM "table";
SELECT MAX("column1") FROM "table";
SELECT MIN("column1") FROM "table";
SELECT SUM("column1") FROM "table";
SELECT COUNT(*) FROM "table";
SELECT DISTINCT "column1" FROM "table";
SELECT COUNT(DISTINCT "column1") FROM "table";

SELECT * FROM "table" 
	WHERE "column1" = (SELECT MAX("column1") FROM "table");
```
# Relating
### IN
The keyword `IN` is used to check whether the desired value is in a given list or set of values.
```sql
-- A query that pulls all the titles of books written by Melchor.
SELECT "title" FROM "books" WHERE "id" IN (
	SELECT "book_id" FROM "authored" WHERE "author_id" = (
		SELECT "id" FROM "authors" WHERE "name" = 'Fernanda Melchor'
	)
);
```
## Joins
### Inner Join
#### JOIN 
We use `JOIN` to take some table and combine it with another table by using the primary key of one table that is a foreign key in another. In SQLite3, the `JOIN` is an inner join, which drops rows that do not have valid matches between the primary key and foreign key. 

As an example, suppose that we have the two tables.
```txt
sea_lions
+-------+-------+------------------------+
|  id   | name  |        species         |
+-------+-------+------------------------+
| 10484 | Ayah  | Zalophus californianus |
| 11728 | Spot  | Zalophus californianus |
| 11729 | Tiger | Zalophus californianus |
| 11732 | Mabel | Zalophus californianus |
| 11734 | Rick  | Zalophus californianus |
| 11790 | Jolee | Zalophus californianus | X id not in migrations
+-------+-------+------------------------+

migrations
+-------+----------+------+
|  id   | distance | days |
+-------+----------+------+
| 10484 | 1000     | 107  |
| 11728 | 1531     | 56   |
| 11729 | 1370     | 37   |
| 11732 | 1622     | 62   |
| 11734 | 1491     | 58   |
| 11735 | 2723     | 82   | X id not in sea_lions
| 11736 | 1571     | 52   | X id not in sea_lions
| 11737 | 1957     | 92   | X id not in sea_lions
+-------+----------+------+
```
We would like to join these two tables so that we have a new table that has information from both tables.
```sql
SELECT * FROM "sea_lions"
	JOIN "migrations" ON "migrations"."id" = "sea_lions"."id";
```
This results in the table below.
```txt
+-------+-------+------------------------+-------+----------+------+
|  id   | name  |        species         |  id   | distance | days |
+-------+-------+------------------------+-------+----------+------+
| 10484 | Ayah  | Zalophus californianus | 10484 | 1000     | 107  |
| 11728 | Spot  | Zalophus californianus | 11728 | 1531     | 56   |
| 11729 | Tiger | Zalophus californianus | 11729 | 1370     | 37   |
| 11732 | Mabel | Zalophus californianus | 11732 | 1622     | 62   |
| 11734 | Rick  | Zalophus californianus | 11734 | 1491     | 58   |
+-------+-------+------------------------+-------+----------+------+
```
#### NATURAL JOIN
If we have two columns in each table that are identically named, then we can use a `NATURAL JOIN` to implicitly assume that we'd like to join on these two columns. Note that we will not get a duplicate id column in this case!
```sql
SELECT * FROM "sea_lions"
NATURAL JOIN "migrations";
```
This natural join results in the table below.
```txt
+-------+-------+------------------------+----------+------+
|  id   | name  |        species         | distance | days |
+-------+-------+------------------------+----------+------+
| 10484 | Ayah  | Zalophus californianus | 1000     | 107  |
| 11728 | Spot  | Zalophus californianus | 1531     | 56   |
| 11729 | Tiger | Zalophus californianus | 1370     | 37   |
| 11732 | Mabel | Zalophus californianus | 1622     | 62   |
| 11734 | Rick  | Zalophus californianus | 1491     | 58   |
+-------+-------+------------------------+----------+------+
```
### Outer Joins
#### LEFT JOIN
A `LEFT JOIN` will prioritize the table on the *left*; that is, the table that you write first.
```sql
SELECT * FROM "sea_lions"
LEFT JOIN "migrations" ON "migrations"."id" = "sea_lions"."id";
```
This results in the table below.
```txt
+-------+-------+------------------------+-------+----------+------+
|  id   | name  |        species         |  id   | distance | days |
+-------+-------+------------------------+-------+----------+------+
| 10484 | Ayah  | Zalophus californianus | 10484 | 1000     | 107  |
| 11728 | Spot  | Zalophus californianus | 11728 | 1531     | 56   |
| 11729 | Tiger | Zalophus californianus | 11729 | 1370     | 37   |
| 11732 | Mabel | Zalophus californianus | 11732 | 1622     | 62   |
| 11734 | Rick  | Zalophus californianus | 11734 | 1491     | 58   |
| 11790 | Jolee | Zalophus californianus |       |          |      |
+-------+-------+------------------------+-------+----------+------+
```
#### RIGHT JOIN
A `RIGHT JOIN` will prioritize the table on the right. 
```sql
SELECT * FROM "sea_lions"
RIGHT JOIN "migrations" ON "migrations"."id" = "sea_lions"."id";
```
Running this query, we get the table below.
```txt
+-------+-------+------------------------+-------+----------+------+
|  id   | name  |        species         |  id   | distance | days |
+-------+-------+------------------------+-------+----------+------+
| 10484 | Ayah  | Zalophus californianus | 10484 | 1000     | 107  |
| 11728 | Spot  | Zalophus californianus | 11728 | 1531     | 56   |
| 11729 | Tiger | Zalophus californianus | 11729 | 1370     | 37   |
| 11732 | Mabel | Zalophus californianus | 11732 | 1622     | 62   |
| 11734 | Rick  | Zalophus californianus | 11734 | 1491     | 58   |
|       |       |                        | 11735 | 2723     | 82   |
|       |       |                        | 11736 | 1571     | 52   |
|       |       |                        | 11737 | 1957     | 92   |
+-------+-------+------------------------+-------+----------+------+
```
#### FULL JOIN
A `FULL JOIN` will let us see the full range of values and which ones are missing.
```sql
SELECT * FROM "sea_lions"
FULL JOIN "migrations" ON "migrations"."id" = "sea_lions"."id";
```
The above query results in the table below.
```txt
+-------+-------+------------------------+-------+----------+------+
|  id   | name  |        species         |  id   | distance | days |
+-------+-------+------------------------+-------+----------+------+
| 10484 | Ayah  | Zalophus californianus | 10484 | 1000     | 107  |
| 11728 | Spot  | Zalophus californianus | 11728 | 1531     | 56   |
| 11729 | Tiger | Zalophus californianus | 11729 | 1370     | 37   |
| 11732 | Mabel | Zalophus californianus | 11732 | 1622     | 62   |
| 11734 | Rick  | Zalophus californianus | 11734 | 1491     | 58   |
| 11790 | Jolee | Zalophus californianus |       |          |      |
|       |       |                        | 11735 | 2723     | 82   |
|       |       |                        | 11736 | 1571     | 52   |
|       |       |                        | 11737 | 1957     | 92   |
+-------+-------+------------------------+-------+----------+------+
```
# Sets
To interact with sets, we use the SQL keywords `UNION`, `INTERSECT`, and `EXCEPT`. For the below example queries, consider two tables containing authors in one and translators in the other.

Note that for these set commands we need to have the same type of columns and rows for each table. 
### UNION
If we'd like to find all translators or authors, we can use `UNION`.
```sql
-- produces a table of all authors and translators
SELECT "name" FROM "translators"
UNION 
SELECT "name" FROM "authors";

-- produces a table of all authors and translators along with 
-- their associated profession. Note this does not consider
-- when an author is also a translator and vice-versa.
SELECT 'author' AS "profession", "name" FROM "authors"
UNION 
SELECT 'translator' AS "profession", "name" FROM "translators";
```
### INTERSECT
We can use `INTERSECT` to find the names who are a translator and an author.
```sql
SELECT "name" FROM "translators"
INTERSECT
SELECT "name" FROM "authors";

-- finds the books that both Sophie Hughes and Margaret Jull Costa
-- have translated.
SELECT "book_id" FROM "translated" WHERE "translator_id" = (
	SELECT "id" FROM "translators" WHERE "name" = 'Sophie Hughes'
)
INTERSECT
SELECT "book_id" FROM "translated" WHERE "translator_id" = (
	SELECT "id" FROM "translators" WHERE "name" = 'Margaret Jull Costa'
)
```
### EXCEPT
We can use `EXCEPT` to find those who are only an author.
```sql
SELECT "name" from "authors"
EXCEPT
SELECT "name" from "translators";
```
# Groups
Suppose we have the table below containing books and their ratings.
```txt
+---------+--------+
| book_id | rating |
+---------+--------+
| 1       | 3      |
| 1       | 5      |
| 1       | 4      |
| 2       | 4      |
| 2       | 4      |
| 2       | 5      |
+---------+--------+
```
### GROUP BY
We can use the `GROUP BY` keyword to group rows according to some column. In this example, we can group each row by `book_id` and then find the average rating of each book.
```sql
SELECT "book_id", ROUND(AVG("rating"), 2) AS "average rating"
FROM "ratings"
GROUP BY "book_id";

SELECT "book_id", COUNT("rating") AS "rating count"
FROM "ratings"
GROUP BY "book_id";
```
### HAVING
The keyword `HAVING` is used similar to `WHERE`, but it is used for groups. If we'd like to select all books that have an average rating greater than 4.0, we can use `HAVING` as below.
```sql
SELECT "book_id", ROUND(AVG("rating"), 2) AS "average rating"
FROM "ratings"
GROUP BY "book_id"
HAVING "average rating" > 4.0;

SELECT "book_id", ROUND(AVG("rating"), 2) AS "average rating"
FROM "ratings"
GROUP BY "book_id"
HAVING "average rating" > 4.0
ORDER BY "average rating" DESC;
```

# Designing
### Type Affinity (Associated to Columns)
Note that columns in SQLite do not always store one particular type. They have type affinities, so they will try to convert an inputted value into the type that they have an affinity for. By default, the type affinity is `NUMERIC` if the type is not specified.
```txt
TEXT
NUMERIC (Stores INTEGER or REAL)
INTEGER
REAL
BLOB
```
### Storage Classes (Associated to Values)
Unlike columns, however, individual values do always have a type within a storage class. The storage classes in SQLite3 are below with their associated data types underneath.
```txt
NULL
INTEGER
	- 0-byte integer        - 4-byte integer
	- 1-byte integer        - 6-byte integer
	- 2-byte integer        - 8-byte integer
	- 3-byte integer
REAL
TEXT
BLOB (binary large object, used for files, images, etc.)
```
Consider the example below.
```txt
						TEXT
				+-------------+
				| id | amount | 
				+-------------+
10 INTEGER	->	| 1  |  "10"  | 
2.5 REAL    ->  | 2  |  "2.5" |
				+-------------+

					  INTEGER
				+-------------+
				| id | amount | 
				+-------------+
-23.45 REAL	->	| 1  |  -23   | REAL is truncated 
2.9 REAL    ->  | 2  |   2    |
				+-------------+
```
### Column Constraints
These are constraints that we can add to columns.
```txt
CHECK
	- Checks if value satifies some condition
DEFAULT
	- If no value specified, column will use default value
NOT NULL
	- Ensures values are not NULL within a column. This is not needed for 
	  primary or foreign keys.
UNIQUE
	- Forces all values to be unique in the column. This is not needed for 
	  primary or foreign keys.
```
### CREATE TABLE
```sql
CREATE TABLE "table1" (
	"column1" [TYPE],
	"column2" [TYPE],
	"column3" [TYPE],
	"column4" [TYPE] [COLUMN CONSTRAINT],
	-- Joint primary key: Every row must have unique column1 and column2
		-- PRIMARY KEY("column1", "column2")
	PRIMARY KEY("column1")
	FOREIGN KEY("column2") REFERENCES "table2"("id"),
	FOREIGN KEY("column3") REFERENCES "table3"("id")
);
```
An example of a `.sql` file for creating a schema is below.
```sql
CREATE TABLE "cards" (
	"id" INTEGER,
	PRIMARY KEY("id")
);
CREATE TABLE "stations" (
	"id" INTEGER,
	"name" TEXT NOT NULL UNIQUE,
	"line" TEXT NOT NULL,
	PRIMARY KEY("id")
);
CREATE TABLE "swipes" (
	"id" INTEGER,
	"card_id" INTEGER,
	"station_id" INTEGER,
	"type" TEXT NOT NULL CHECK("type" IN ('enter', 'exit', 'deposit')),
	"datetime" NUMERIC NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"amount" NUMERIC NOT NULL CHECK("amount" != 0),
	FOREIGN KEY("card_id") REFERENCES "cards"("id"),
	FOREIGN KEY("station_id") REFERENCES "stations"("id")
);
```
### DROP TABLE
To drop a table, you use the keyword `DROP TABLE`.
```sql
DROP TABLE "table";
```
### ALTER TABLE
```sql
ALTER TABLE "table"
ADD COLUMN "column";

ALTER TABLE "table"
RENAME TO "new_table";

ALTER TABLE "table"
RENAME COLUMN "column" TO "new_column";

ALTER TABLE "table"
DROP COLUMN "column" [TYPE];
```
# Writing
Note that to import a CSV file into a SQLite3 database, we can use the command `.import`. We can use this as `.import --csv [file] [table]`. To ignore the first row, we can also use the `--skip 1` flag.
### INSERT
```sql
INSERT INTO "collections" ("title", "accession_number", "acquired")
VALUES ('Profusion of Flowers', '56.257', '1956-04-12');

INSERT INTO "collections" ("title", "accession_number", "acquired") 
VALUES 
('Imaginative landscape', '56.496', NULL),
('Peonies and butterfly', '06.1899', '1906-01-01');

INSERT INTO "collections" ("title", "accession_number", "acquired") 
SELECT "title", "accession_number", "acquired" FROM "temp";
```
### DELETE
```sql
DELETE FROM "table";

DELETE FROM "collections"
WHERE "acquired" < '1909-01-01';
```
### Foreign Key Constraints
The foreign key constraints specify actions to be taken when an ID referenced by a foreign key is deleted. We use the keyword `ON DELETE` followed by one of the below actions.
```txt
RESTRICT
NO ACTION
SET NULL
SET DEFAULT
CASCADE
```
An example using `CASCADE` is below.
```sql
FOREIGN KEY("artist_id") REFERENCES "artists"("id") ON DELETE CASCADE
```
### UPDATE
```sql
UPDATE "table" SET "column0" = [VALUE0], ..., "columnN" = [VALUEN]
WHERE condition;

-- Examples
UPDATE "created" SET "artist_id" = 3
WHERE "collection_id" = (
	SELECT "id" FROM "collections"
	WHERE "title" = 'Farmers working at dawn'
);
UPDATE "votes" SET "title" = 'FARMERS WORKING AT DAWN'
WHERE "title" LIKE 'fa%'; -- NOTE: LIKE is case-insensitive
```
### Scalar Functions in SQLite3
These are functions that take in a value and return a different value within SQLite3.
```sql
-- trim() is SQLite3 function that removes white space on edges of text
UPDATE "votes" SET "title" = trim("title")
-- upper() turns all text strings into uppercase
UPDATE "votes" SET "title" = upper("title")
-- lower() turns all text strings into lowercase
UPDATE "votes" SET "title" = lower("title")
-- There are many more! (https://www.sqlite.org/lang_corefunc.html)
```
### TRIGGER
A trigger will listen to SQL keywords used on a table and react automatically to them. For example, we could have a trigger on a table that listens for a `DELETE` keyword and subsequently inserts the deleted row into a different table. 
```sql
CREATE TRIGGER name
[AFTER|BEFORE] [INSERT ON|UPDATE OF [column] ON|DELETE ON] table
FOR EACH ROW
BEGIN
[STATEMENT TO RUN];
END;

-- Example
-- We would like a row to be automatically added to the "transactions" table
-- when a painting is removed from the "collections" table
CREATE TRIGGER "sell"
BEFORE DELETE ON "collections"
FOR EACH ROW
BEGIN
	INSERT INTO "transactions" ("title", "action")
	VALUES (OLD."title", 'sold'); -- OLD gives access to old row 
END;
-- We would like to implement a similar trigger, but for purchasing and adding
-- paintings to our collection
CREATE TRIGGER "buy"
AFTER INSERT on "collections"
FOR EACH ROW
BEGIN
	INSERT INTO "transactions" ("title", "actions")
	VALUES (NEW."title", 'bought'); -- NEW gives access to new row
END;
```
### Soft Deletions
This is a technique to store deleted entries. Instead of deleting an item, we can have a column `deleted`, which defaults to 0, and can change the value to 1 when we delete that row.
