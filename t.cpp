#include <iostream>
#include <algorithm>
using namespace std;

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n;cin>>n;
        int a[n];
        for (int i = 0; i < n; i++)
        {
            cin>>a[i];
        }
        int l =0;
        int r =n-1;
        string s;
        bool flag = true;
        while(l<=r)
        {
           if(flag)
           {
            if(a[l]>a[r])
            {
                s.push_back('R');
                r--;
            }
            else{
                s.push_back('L');
                l++;
            }
            flag = false;
           }
           else{
              if(a[l]>a[r])
            {
                s.push_back('R');
                l++;
            }
            else{
                s.push_back('L');
                r--;
            }
            flag = true;
           }
        }
        cout<<s<<endl;
    }

    return 0;
}