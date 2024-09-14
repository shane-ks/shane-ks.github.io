# Types of Relationships
Consider an example of a database that stores authors and the books they have written. There are several ways we can choose to organize our data.
```txt
One author to one book.     --> One-to-One Relationship
One author to many books.   --> One-to-Many Relationship
Many authors to many books. --> Many-to-Many Relationship
```
# Entity Relationship (ER) Diagram
ER Diagrams are used to understand how tables are related to each other within a database. An example of one (credit to the CS50 SQL course) is provided below.
![[er_diagram.png|400]]
The lines connecting the squares are in crow's foot notation, which is shown below. 
![[er_crows_feet.png|200]]
# Keys
There are two types of keys. A ==Primary Key== is an identifier that is unique for every item in a table. On the other hand, a ==Foreign Key== is when we take a primary key from one table and include it in a column of a different table. 

For example, consider two tables of books and ratings. We have an ISBN for each book, which we are using as a primary key in the books table. We can use the ISBN as a foreign key inside of the ratings table so that we can tie each rating to a specific book. So, we will have unique ISBNs in the books table, but many ratings per ISBN in the ratings table.
# Subqueries (Nested Queries)
```sql
SELECT "name" FROM "authors"
WHERE "id" = (
	SELECT "author_id" FROM "authored"
	WHERE "book_id" = (
		SELECT "id" FROM "books"
		WHERE "title" = "book_name"
	)
)
```

# IN
The keyword `IN` is used to check whether the desired value is in a given list or set of values.