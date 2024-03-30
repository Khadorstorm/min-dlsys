

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

#### Gradient for sum & broadcast (3.30)

##### Sum:

The idea is simple, we broadcast the out_grad in the dimension(**s**) of summation since with sum every $\frac{\partial{y}}{\partial{x}}$ is 1.

To implement it is a bit harder for outgrad can be 1d array sometimes and we can only use needle functions instead of the lovely `[:, np.newaxis]` to reshape it.

**Solution:** get the shape of node input and manually change the summed dimension-th elements to 1, then `reshape()` the `outgrad` using this new correct shape list.

##### Broadcast:

Counterintuitive to understand and painful to implement 💀 

The gradient of each element in the input should be the sum of all the gradients that involve the broadcasted selves in `outgrad` (not sure why but that's the only idea I could come up with and it happens to be correct :P). Maybe that's for when one varible is used as input for multiple computation? Need to find some examples using broadcast in models to understand this better.

To implement this is to find the axes to sum with.

If the input and outgrad have the same amount of dimensions  *e.g.(2,3,1) and (2,3,4)* we just simply get the unequal dimensions.

Things get ugly when the input is a 1darray **(n,)** or a number **()**,  my solution is to patch some extra dimensions to it first and follow the same idea. *e.g.(2,)->(2,1,1)*

And after summation we need to reshape the result to the input shape so it won't be missing dimensions that are 1.

#### `matmul` behaviors

- If both arguments are 2-D they are multiplied like conventional matrices.
- If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

Therefore, when one of the input is more than 2d, we need to sum up the extra dimensions in result  gradient

e.g. a.shape= (6,6,5,4) matmul b.shape=(4,3)

after a naive calculation grad_b is now (6,6,4,3), we need to sum up the first 2 dimensions.

#### List["Tensor"]

- `List[Tensor]` is a straightforward type hint that requires the `Tensor` class to be available at the point of definition.
- `List["Tensor"]` is a forward-declared type hint that allows the `Tensor` class to be defined later in the code, helping to avoid circular import issues.
- In Python 3.7 and above, you can use the `from __future__ import annotations` import to automatically treat all annotations as forward declarations, removing the need to quote type hints.

#### Topological sort

Recrusive function: for every node, deal with all its inputs first, then add the node to the sort results. 

Input order of nodes does not matter

Need to maintain a `visited` Set