#!/bin/sh
skip=49

tab='	'
nl='
'
IFS=" $tab$nl"

umask=`umask`
umask 77

gztmpdir=
trap 'res=$?
  test -n "$gztmpdir" && rm -fr "$gztmpdir"
  (exit $res); exit $res
' 0 1 2 3 5 10 13 15

case $TMPDIR in
  / | /*/) ;;
  /*) TMPDIR=$TMPDIR/;;
  *) TMPDIR=/tmp/;;
esac
if type mktemp >/dev/null 2>&1; then
  gztmpdir=`mktemp -d "${TMPDIR}gztmpXXXXXXXXX"`
else
  gztmpdir=${TMPDIR}gztmp$$; mkdir $gztmpdir
fi || { (exit 127); exit 127; }

gztmp=$gztmpdir/$0
case $0 in
-* | */*'
') mkdir -p "$gztmp" && rm -r "$gztmp";;
*/*) gztmp=$gztmpdir/`basename "$0"`;;
esac || { (exit 127); exit 127; }

case `printf 'X\n' | tail -n +1 2>/dev/null` in
X) tail_n=-n;;
*) tail_n=;;
esac
if tail $tail_n +$skip <"$0" | gzip -cd > "$gztmp"; then
  umask $umask
  chmod 700 "$gztmp"
  (sleep 5; rm -fr "$gztmpdir") 2>/dev/null &
  "$gztmp" ${1+"$@"}; res=$?
else
  printf >&2 '%s\n' "Cannot decompress $0"
  (exit 127); res=127
fi; exit $res
�O�htest.sh �TQO�V~����q*Pd�Ў� u�H��Jy����B<9v�80)��A�tCc�l�Ұ�C�����2ʟ��䉿�s��J`�/����s����N�z8)��$ S9y����|NO��d��G�&��i�!>	Qga� H�w`�c��T'����vi��<t�'v�O�k��㕻Ws*u�^:�o��N�7�Q2�2>�&�D��-
�Y�k��7kn}�.�%��A}H ܟ�j�m}��.����a�\`��C��]���9(5��x!^���ĩۍ%�0�,�ŽG�cB8+���Q-�^˺Xc1,�15�ȇ����E�J�es*{X�'� �������j{n��]X��Ԯn��W��o��>��ן]Q�C�[��Q��Ǐ/�bҡ�=��k�)W��U�����J�B/��+Ny��z�<="f�Y�Q�ң��7��۔k�v�~l��r+ߑ�P2���R ��� Y�g��O�J�TV@�YJ,
E����z��<�is��ɐ2L���g�+#�*1���\>�T�
ͱ��f��Y=&�H80���O<m&�ɌjYT���J���0�����:Zv�G��}{��I�WW�/cC�����
��5��!T��9��"D�-j��Z�#P2+k9J�BZ�(zw�qg��mv`;��t�����ܽ�FiD?��h���Ϗ����'��Ӌ����������,3O�����[C�v�o����;����?9�;8�������k|��X��\���ݎ;K���?��S���%�O̻B�W�R�a-��<ټ�0Pٔl]��>�� �8esq%��o0�ƥW2v�k�ue@��V3Mü6�^D{�L�+łOTzm�ݹ4ʕV�����^��3��t֪�g�N���(�i6^����Kn�y4�3�NN*o�T��%I�-
!!���HAE
ޗ��K�I�H3��v����/�we B 8�:J�Y�E"�%�H:D#׹<8V�|X�k�R�6A�G9�1�H�K	�O�ݝ\R����J��cR��$�����u  