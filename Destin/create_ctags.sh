#creates ctags to be used by vim.
#ignores files that start with an underscore
find . ! -path "*/CMakeFiles*" -a  -regex ".*\.\(c\|cpp\|h\)" -a ! -name "_*" | xargs ctags --defines 

