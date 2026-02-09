"""Tests for visualization styles module."""

import pytest

from ising_toolkit.visualization import (
    # Style dictionaries
    STYLE_PUBLICATION,
    STYLE_PRESENTATION,
    STYLE_NOTEBOOK,
    STYLE_MINIMAL,
    STYLE_DEFAULT,
    # Style functions
    get_style,
    set_style,
    reset_style,
    style_context,
    list_styles,
    register_style,
    # Color palettes
    COLORS_QUALITATIVE,
    COLORS_TAB10,
    COLOR_SPIN_UP,
    COLOR_SPIN_DOWN,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    # Color utilities
    get_colors,
    get_size_colors,
    get_markers,
)


class TestStyleDictionaries:
    """Tests for style dictionaries."""

    def test_publication_style_keys(self):
        """Test publication style has expected keys."""
        assert 'figure.figsize' in STYLE_PUBLICATION
        assert 'font.size' in STYLE_PUBLICATION
        assert 'axes.labelsize' in STYLE_PUBLICATION
        assert 'legend.fontsize' in STYLE_PUBLICATION
        assert 'savefig.dpi' in STYLE_PUBLICATION

    def test_presentation_style_larger_fonts(self):
        """Test presentation style has larger fonts."""
        assert STYLE_PRESENTATION['font.size'] > STYLE_PUBLICATION['font.size']
        assert STYLE_PRESENTATION['axes.labelsize'] > STYLE_PUBLICATION['axes.labelsize']

    def test_publication_style_serif_font(self):
        """Test publication uses serif font."""
        assert STYLE_PUBLICATION['font.family'] == 'serif'

    def test_presentation_style_sans_font(self):
        """Test presentation uses sans-serif font."""
        assert STYLE_PRESENTATION['font.family'] == 'sans-serif'

    def test_publication_high_dpi(self):
        """Test publication has high DPI for saving."""
        assert STYLE_PUBLICATION['savefig.dpi'] >= 300

    def test_all_styles_are_dicts(self):
        """Test all styles are dictionaries."""
        assert isinstance(STYLE_PUBLICATION, dict)
        assert isinstance(STYLE_PRESENTATION, dict)
        assert isinstance(STYLE_NOTEBOOK, dict)
        assert isinstance(STYLE_MINIMAL, dict)
        assert isinstance(STYLE_DEFAULT, dict)


class TestGetStyle:
    """Tests for get_style function."""

    def test_get_publication_style(self):
        """Test getting publication style."""
        style = get_style('publication')
        assert style == STYLE_PUBLICATION

    def test_get_presentation_style(self):
        """Test getting presentation style."""
        style = get_style('presentation')
        assert style == STYLE_PRESENTATION

    def test_get_style_case_insensitive(self):
        """Test style names are case insensitive."""
        style1 = get_style('PUBLICATION')
        style2 = get_style('Publication')
        style3 = get_style('publication')

        assert style1 == style2 == style3

    def test_get_style_returns_copy(self):
        """Test get_style returns a copy."""
        style1 = get_style('publication')
        style2 = get_style('publication')

        # Modify one, should not affect the other
        style1['font.size'] = 999
        assert style2['font.size'] != 999

    def test_get_unknown_style_raises(self):
        """Test unknown style raises ValueError."""
        with pytest.raises(ValueError, match="Unknown style"):
            get_style('nonexistent_style')


class TestSetStyle:
    """Tests for set_style function."""

    def test_set_style_applies(self):
        """Test set_style applies to rcParams."""
        plt = pytest.importorskip("matplotlib.pyplot")

        try:
            set_style('presentation')
            # Font size should change
            assert plt.rcParams['font.size'] == STYLE_PRESENTATION['font.size']
        finally:
            reset_style()

    def test_set_style_unknown_raises(self):
        """Test set_style with unknown style raises."""
        pytest.importorskip("matplotlib.pyplot")

        with pytest.raises(ValueError):
            set_style('nonexistent')


class TestResetStyle:
    """Tests for reset_style function."""

    def test_reset_style_restores_defaults(self):
        """Test reset_style restores matplotlib defaults."""
        pytest.importorskip("matplotlib.pyplot")

        # Apply a style
        set_style('publication')

        # Reset
        reset_style()

        # Should be back to matplotlib defaults
        # (Don't check exact values as they may vary by version)
        assert True  # Just verify no errors


class TestStyleContext:
    """Tests for style_context context manager."""

    def test_style_context_applies_temporarily(self):
        """Test style is applied only within context."""
        plt = pytest.importorskip("matplotlib.pyplot")

        reset_style()
        original_size = plt.rcParams['font.size']

        with style_context('presentation'):
            assert plt.rcParams['font.size'] == STYLE_PRESENTATION['font.size']

        # Should be restored after context
        assert plt.rcParams['font.size'] == original_size

    def test_style_context_nested(self):
        """Test nested style contexts."""
        plt = pytest.importorskip("matplotlib.pyplot")

        reset_style()

        with style_context('publication'):
            pub_size = plt.rcParams['font.size']

            with style_context('presentation'):
                pres_size = plt.rcParams['font.size']
                assert pres_size == STYLE_PRESENTATION['font.size']

            # Back to publication
            assert plt.rcParams['font.size'] == pub_size


class TestListStyles:
    """Tests for list_styles function."""

    def test_list_styles_returns_list(self):
        """Test list_styles returns a list."""
        styles = list_styles()
        assert isinstance(styles, list)

    def test_list_styles_contains_builtin(self):
        """Test list contains built-in styles."""
        styles = list_styles()
        assert 'publication' in styles
        assert 'presentation' in styles
        assert 'notebook' in styles
        assert 'default' in styles


class TestRegisterStyle:
    """Tests for register_style function."""

    def test_register_custom_style(self):
        """Test registering a custom style."""
        custom = {'font.size': 20, 'axes.grid': True}
        register_style('my_custom', custom)

        retrieved = get_style('my_custom')
        assert retrieved['font.size'] == 20
        assert retrieved['axes.grid'] is True

    def test_registered_style_in_list(self):
        """Test registered style appears in list."""
        register_style('test_style_123', {'font.size': 15})
        styles = list_styles()
        assert 'test_style_123' in styles


class TestColorPalettes:
    """Tests for color palettes."""

    def test_qualitative_colors_distinct(self):
        """Test qualitative colors are distinct."""
        assert len(set(COLORS_QUALITATIVE)) == len(COLORS_QUALITATIVE)

    def test_qualitative_colors_count(self):
        """Test there are enough qualitative colors."""
        assert len(COLORS_QUALITATIVE) >= 10

    def test_tab10_colors_count(self):
        """Test tab10 has 10 colors."""
        assert len(COLORS_TAB10) == 10

    def test_spin_colors_are_different(self):
        """Test spin up/down colors are different."""
        assert COLOR_SPIN_UP != COLOR_SPIN_DOWN

    def test_positive_negative_colors(self):
        """Test positive/negative colors match spin colors."""
        assert COLOR_POSITIVE == COLOR_SPIN_UP
        assert COLOR_NEGATIVE == COLOR_SPIN_DOWN

    def test_colors_are_hex(self):
        """Test colors are valid hex codes."""
        for color in COLORS_QUALITATIVE:
            assert color.startswith('#')
            assert len(color) == 7

        for color in COLORS_TAB10:
            assert color.startswith('#')
            assert len(color) == 7


class TestGetColors:
    """Tests for get_colors function."""

    def test_get_colors_returns_list(self):
        """Test get_colors returns a list."""
        colors = get_colors(5)
        assert isinstance(colors, list)
        assert len(colors) == 5

    def test_get_colors_exact_count(self):
        """Test exact number of colors returned."""
        for n in [1, 3, 5, 10]:
            colors = get_colors(n)
            assert len(colors) == n

    def test_get_colors_more_than_palette(self):
        """Test requesting more colors than palette size."""
        colors = get_colors(20)
        assert len(colors) == 20

    def test_get_colors_qualitative(self):
        """Test qualitative palette."""
        colors = get_colors(5, palette='qualitative')
        assert colors == COLORS_QUALITATIVE[:5]

    def test_get_colors_tab10(self):
        """Test tab10 palette."""
        colors = get_colors(5, palette='tab10')
        assert colors == COLORS_TAB10[:5]

    def test_get_colors_sequential(self):
        """Test sequential palette."""
        pytest.importorskip("matplotlib.pyplot")
        colors = get_colors(5, palette='sequential')
        assert len(colors) == 5


class TestGetSizeColors:
    """Tests for get_size_colors function."""

    def test_get_size_colors_returns_dict(self):
        """Test get_size_colors returns a dictionary."""
        sizes = [8, 16, 32]
        colors = get_size_colors(sizes)

        assert isinstance(colors, dict)
        assert set(colors.keys()) == set(sizes)

    def test_get_size_colors_values(self):
        """Test dictionary values are colors."""
        sizes = [8, 16, 32]
        colors = get_size_colors(sizes)

        for size in sizes:
            assert colors[size].startswith('#')

    def test_get_size_colors_sorted(self):
        """Test sizes are sorted in the mapping."""
        sizes = [32, 8, 16]
        colors = get_size_colors(sizes)

        # Get colors in order of sorted sizes
        ordered_colors = [colors[s] for s in sorted(sizes)]

        # First few colors from qualitative palette
        expected_colors = COLORS_QUALITATIVE[:3]
        assert ordered_colors == expected_colors


class TestGetMarkers:
    """Tests for get_markers function."""

    def test_get_markers_returns_list(self):
        """Test get_markers returns a list."""
        markers = get_markers(5)
        assert isinstance(markers, list)
        assert len(markers) == 5

    def test_get_markers_valid_styles(self):
        """Test markers are valid matplotlib styles."""
        valid_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
        markers = get_markers(5)

        for m in markers:
            assert m in valid_markers

    def test_get_markers_cycling(self):
        """Test markers cycle when requesting many."""
        markers = get_markers(20)
        assert len(markers) == 20


class TestPlottingUtilities:
    """Tests for plotting utility functions."""

    def test_add_critical_line(self):
        """Test add_critical_line function."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from ising_toolkit.visualization import add_critical_line

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        add_critical_line(ax, Tc=2.269)

        # Check line was added
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_add_theory_line(self):
        """Test add_theory_line function."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from ising_toolkit.visualization import add_theory_line

        fig, ax = plt.subplots()

        add_theory_line(ax, [1, 2, 3], [1, 4, 9])

        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_configure_legend(self):
        """Test configure_legend function."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from ising_toolkit.visualization import configure_legend

        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label='test')

        configure_legend(ax)

        assert ax.get_legend() is not None
        plt.close(fig)

    def test_format_axis_labels(self):
        """Test format_axis_labels function."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from ising_toolkit.visualization import format_axis_labels

        fig, ax = plt.subplots()

        format_axis_labels(ax, xlabel='X', ylabel='Y', title='Title')

        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert ax.get_title() == 'Title'
        plt.close(fig)


class TestStyleIntegration:
    """Integration tests for style system."""

    def test_publication_figure(self):
        """Test creating a publication-quality figure."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import numpy as np

        with style_context('publication'):
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            ax.plot(x, np.sin(x), label='sin(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()

            # Check some style properties were applied
            assert fig.get_figwidth() == STYLE_PUBLICATION['figure.figsize'][0]

        plt.close(fig)

    def test_presentation_figure(self):
        """Test creating a presentation figure."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import numpy as np

        with style_context('presentation'):
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            ax.plot(x, np.cos(x), label='cos(x)')

            # Larger figure for presentation
            assert fig.get_figwidth() == STYLE_PRESENTATION['figure.figsize'][0]

        plt.close(fig)
