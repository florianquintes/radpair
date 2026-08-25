# radpair

`radpair` is a Python package for the analytic simulation of continuous-wave
electron paramagnetic resonance (cw-EPR) spectra of singlet-born
spin-correlated radical pairs.

The package solves the spin Hamiltonian analytically using a pseudo-secular
approximation for the hyperfine couplings, enabling fast spectral computation
without numerical diagonalisation.

## Features

- Analytic solution of the spin Hamiltonian for radical pairs
- Pseudo-secular approximation for hyperfine couplings
- Support for up to five anisotropic nuclei groups (donor or acceptor)
- Zero-field splitting (ZFS) tensor with parameters *D* and *E*
- Exchange interaction *J* and anisotropic *g*-tensors
- Optional interpolation for refined orientation grids
- Single-core and multi-core execution via `multiprocessing.Pool`

## Requirements

- Python 3.13 or newer
- `eprbase`, `NumPy`, and `SciPy`

## Installation

### Using uv

```bash
uv add radpair
```

### Using pip

```bash
pip install radpair
```

## Development

Clone the repository and install the development dependencies with `uv`:

```bash
git clone https://github.com/florianquintes/radpair.git
cd radpair
uv sync --dev
```

Run the tests:

```bash
uv run pytest
```

## Documentation

The documentation is available on
[GitHub Pages](https://florianquintes.github.io/radpair/).

## License

This project is licensed under the GNU General Public License v3.0.
See the [LICENSE](LICENSE) file for details.
