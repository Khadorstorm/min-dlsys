

## Min Journal

### Hw1 

#### `transpose` & `swapaxes` (2024.3.29)

`transpose` change multiple dimensions based on the given `axes` sequence. So if you're doing strange stuff you should give the whole new order of axes.

`swapaxes` on the other hand just exchange 2 axis given

so in this case we can not use `transpose` to implement the needle Transpose because things like this will happen:

```python
# Incorrect usage leading to error or unexpected result
try:
    transposed = np.transpose(a, axes=(1, 0))  # This is incomplete and will cause an error
except ValueError as e:
    print(e)

# Correct usage
transposed = np.transpose(a, axes=(1, 0, 2))
print(transposed.shape)  # Output will be (4, 3, 5)

```

That's why we got a ValueError.