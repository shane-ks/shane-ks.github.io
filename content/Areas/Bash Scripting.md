# Variables
We can define a variable in the shell by typing `foo=bar` and then access `foo` by `$foo`. We can define strings in Bash with `""` and `''`. There are subtle differences between these as shown below.

```bash
foo=bar
echo $foo   ->   "bar"
echo "Value is $foo"   ->   "Value is bar"
echo 'Value is $foo'   ->   "Value is $foo"
```
# Functions
Function are defined as below. We can access the first argument by `$1` and similarly for the nth argument. The `$0` variable is the name of the script. 
```bash
mcd () {
	mkdir -p "$1"
	cd "$1"
}
```
The function can then be loaded into the shell by `source [file]`. 
```bash
$0 -> "Name of the function"
$1 -> "The first argument"
$[N] -> "The N-th argument" 
$? -> "The error code of the previous command"
$_ -> "Gives the last argument to the previous command"
$# -> "The number of arguments"
$@ -> "Expands to all of the arguments. Can be used with for loop"
$(date) -> "The current date"
$$ -> "The current PID"
```
# Redirection
```bash
[command] < [file]  -> "Maps STDIN of command to the file"
[command] > [file]  -> "Maps STDOUT of command to the file"
[command] >> [file] -> "Maps STDOUT of command to the file BUT only appends."
[command] 2> [file] -> "Maps STDERR of command to the file"
```
# Logical Operators
```bash
[command1] || [command2] -> "command2 will run if command1 has a 1 error code."
[command1] && [command2] -> "command2 will run only if command1 has 0 error                                  code."
```
# Loops and Conditionals
```bash
for file in "$@"; do
	grep foobar "$file" > /dev/null 2> /dev/null
	if [["$?" -eq 1]]; then
		echo "File $file does not have any foobar, so appending one"
		echo "# foobar" >> "$file"
	elif [["$?" -eq 0]]; then
		echo "Found a foobar!"
	else 
		echo "Error code different from 0 and 1."
	fi
done 
``` 
# Case Statements
The case statement can be used to match on values. 
```bash
day=$(date + "%a")

case $day in 
	Mon | Tues | Wed | Thu | Fri)
		echo "Today is a weekday."
		;;
	Sat | Sun)
		echo "Today is the weekend."
		;;
	*) 
		echo "Day not recognized."
		;;
esac
```
# Misc.
```bash
!! -> "Inserts the last ran command"
[command1] ; [command2] -> "runs command1 and then command2"
<([command]) -> "places the output of [command] in a temporary file. This is                      useful for command that expect inputs to be files."
			 -> cat <(ls) <(ls ..) -> "outputs cat of ls in current and prev dir"
/dev/null -> "Can write here to discard outputs."
man test -> "The man page of all the comparison operators in bash."
*.sh -> "* is a wildcard so matches with any number of characters."
test.s? -> "? matches with any singular character."
test{,1,2,3} -> "Expands into" test test1 test2 test3
test{a..c} -> "Expands into" testa testb testc
#!/usr/bin/env python -> Place this at the top of a script to run .py scripts

```