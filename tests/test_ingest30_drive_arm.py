import pytest

from isaac_datagen.ingest30_drive import (
    ARM_BOOL_FLAGS, ARM_VALUE_FLAGS, arm_commands, parse_args,
)


def flat(cmd):
    return " ".join(str(x) for x in (cmd.argv if hasattr(cmd, "argv") else cmd))


def test_retrained_emits_retrain_loop_with_sizes():
    (cmd,) = arm_commands("R", "retrained", "base.yaml", None, None, "10,20,30", False)
    s = flat(cmd)
    assert "ingest30-retrain-loop" in s
    assert "--sizes 10,20,30" in s and "--label retrained" in s
    assert "--all-data" not in s


def test_retrained_requires_sizes():
    with pytest.raises(AssertionError, match="--sizes"):
        arm_commands("R", "retrained", "base.yaml", None, None, None, False)


def test_retrained_rejects_second_config():
    with pytest.raises(AssertionError, match="ONE config"):
        arm_commands("R", "retrained", "base.yaml", "ingest.yaml", None, "10", False)


def test_non_retrained_arm_rejects_sizes():
    with pytest.raises(AssertionError, match="retrained arm"):
        arm_commands("R", "gligen", "b.yaml", "i.yaml", None, "10", False)


def test_gligen_arm_still_emits_base_train_then_loop():
    cmds = arm_commands("R", "gligen", "b.yaml", "i.yaml", None, None, False)
    assert "ingest30-base-train" in flat(cmds[0])
    assert "ingest30-loop" in flat(cmds[1])


def test_sizes_is_registered_as_a_value_flag():
    assert "--sizes" in ARM_VALUE_FLAGS      # takes a value
    assert "--sizes" not in ARM_BOOL_FLAGS
    assert "--all-data" not in ARM_BOOL_FLAGS


def test_interspersed_sizes_parses_the_canonical_form():
    # the spec's canonical CLI: flag sits between the two positionals
    a = parse_args(["arm", "retrained", "r.yaml", "--sizes", "10,20,30", "/r"])
    assert a.sizes == "10,20,30" and a.label is None


def test_trailing_sizes_also_parses():
    a = parse_args(["arm", "retrained", "r.yaml", "/r", "--sizes", "10,20,30"])
    assert a.sizes == "10,20,30"
